{
  Export selected quest dialogues for Fallout 4
  Apply to quest record(s)
}
unit ExportDialogue;

const
  fDebug = True;

var
  slExport: TStringList;
  lstRecursion: TList;
  ExportFileName, InfoNPCID, InfoSPEAKER, InfoRACEID, InfoCONDITION: string;

//============================================================================
// uncomment to check debug messages
procedure AddDebug(aMsg: string);
begin
  if fDebug then
    AddMessage('DEBUG: ' + aMsg);
end;

//============================================================================
function Initialize: integer;
begin
  slExport := TStringList.Create;
  lstRecursion := TList.Create;
end;

//============================================================================
function InfoFileName(InfoFormID, RespNum: integer): string;
begin
  // direct string-concat avoids any Format() pitfalls
  Result := IntToHex(InfoFormID and $FFFFFF, 8) + '_' + IntToStr(RespNum);
end;

//============================================================================
// CSV field escaping function
function EscapeCSVField(const s: string): string;
begin
  Result := s;
  if (Pos('"', Result) > 0) or (Pos(',', Result) > 0) or (Pos(#10, Result) > 0) or (Pos(#13, Result) > 0) then
  begin
    Result := StringReplace(Result, '"', '""', [rfReplaceAll]);
    Result := '"' + Result + '"';
  end;
end;

//============================================================================
// get voice types from record or reference of record (not to be called directly)
procedure GetRecordVoiceTypes2(e: IInterface; lstVoice: TStringList);
var
  sig: string;
  i: integer;
  ent, ents: IInterface;
begin
  // recursive function, prevent calling itself in loop
  // check against a list of processed FormIDs
  if lstRecursion.IndexOf(FormID(e)) <> -1 then Exit
    else lstRecursion.Add(FormID(e));
  
  e := WinningOverride(e);
  AddDebug('GetRecordVoiceType '+Name(e));

  sig := Signature(e);
  if (sig = 'REFR') or (sig = 'ACHR') then begin
    e := WinningOverride(BaseRecord(e));
    sig := Signature(e);
  end;
    
  // Voice type record
  if sig = 'VTYP' then begin
    lstVoice.AddObject(EditorID(e), e);
  end
  // NPC record
  else if sig = 'NPC_' then begin
    // if has voice then get it
    if ElementExists(e, 'VTCK - Voice') then begin
      InfoNPCID := Name(e);
      InfoRACEID := GetElementEditValues(e, 'RNAM - Race');
      ent := LinksTo(ElementByName(e, 'VTCK - Voice'));
      GetRecordVoiceTypes2(ent, lstVoice);
    end
    // otherwise get voice from template
    else if ElementExists(e, 'TPLT - Template') then begin
      ent := LinksTo(ElementByName(e, 'TPLT - Template'));
      GetRecordVoiceTypes2(ent, lstVoice);
    end;
  end
  // npc leveled list
  else if sig = 'LVLN' then begin
    ents := ElementByName(e, 'Leveled List Entries');
    for i := 0 to Pred(ElementCount(ents)) do begin
      ent := ElementByIndex(ents, i);
      ent := LinksTo(ElementByPath(ent, 'LVLO\Reference'));
      GetRecordVoiceTypes2(ent, lstVoice);
    end;
  end
  // Talking activator record
  else if sig = 'TACT' then begin
    ent := WinningOverride(LinksTo(ElementByName(e, 'VNAM - Voice Type')));
    if Signature(ent) = 'VTYP' then lstVoice.AddObject(EditorID(ent), ent);
  end
  // Form List record
  else if sig = 'FLST' then begin
    ents := ElementByName(e, 'FormIDs');
    for i := 0 to Pred(ElementCount(ents)) do begin
      ent := LinksTo(ElementByIndex(ents, i));
      GetRecordVoiceTypes2(ent, lstVoice);
    end;
  end
  // Faction/class record
  else if (sig = 'FACT') or (sig = 'CLAS') then begin
    for i := 0 to Pred(ReferencedByCount(e)) do begin
      ent := ReferencedByIndex(e, i);
      if Signature(ent) = 'NPC_' then
        GetRecordVoiceTypes2(ent, lstVoice);
    end;
  end;
end;

//============================================================================
// get voice types from record or reference of record
procedure GetRecordVoiceTypes(e: IInterface; lstVoice: TStringList);
begin
  // clear recursion list
  lstRecursion.Clear;
  GetRecordVoiceTypes2(e, lstVoice);
end;

//============================================================================
// get voice types from quest alias
procedure GetAliasVoiceTypes(Quest: IInterface; aAlias: integer; lstVoice: TStringList);
var
  Aliases, Alias: IInterface;
  i, j: integer;
  lstLimit: TStringList;
begin
  Quest := WinningOverride(Quest);
  AddDebug('GetAliasVoiceTypes ' + Name(Quest) + ' -- ' + IntToStr(aAlias));
  
  Aliases := ElementByName(Quest, 'Aliases');
  for i := 0 to Pred(ElementCount(Aliases)) do begin
    Alias := ElementByIndex(Aliases, i);
    if GetNativeValue(ElementByIndex(Alias, 0)) <> aAlias then
      Continue;

    if ElementExists(Alias, 'ALFR - Forced Reference') then
      GetRecordVoiceTypes(LinksTo(ElementByName(Alias, 'ALFR - Forced Reference')), lstVoice)
    else if ElementExists(Alias, 'ALUA - Unique Actor') then
      GetRecordVoiceTypes(LinksTo(ElementByName(Alias, 'ALUA - Unique Actor')), lstVoice)
    else if ElementExists(Alias, 'External Alias Reference') then
      GetAliasVoiceTypes(
        LinksTo(ElementByPath(Alias, 'External Alias Reference\ALEQ - Quest')),
        GetElementNativeValues(Alias, 'External Alias Reference\ALEA - Alias'),
        lstVoice
      )
    else if ElementExists(Alias, 'Conditions') then
      GetConditionsVoiceTypes(ElementByName(Alias, 'Conditions'), lstVoice);
    
    // from CK Wiki "... list of voice types which the editor uses to limit any dialogue assigned to this alias"
    // so remove everything that is not in list
    if GetElementNativeValues(Alias, 'VTCK - Voice Types') <> 0 then begin
      // if we already have voices, then use it as a limit
      if lstVoice.Count <> 0 then begin
        lstLimit := TStringList.Create;
        GetRecordVoiceTypes(LinksTo(ElementByName(Alias, 'VTCK - Voice Types')), lstLimit);
        for j := Pred(lstVoice.Count) downto 0 do
          if lstLimit.IndexOf(lstVoice[j]) = -1 then
            lstVoice.Delete(j);
        lstLimit.Free;
      end
      // otherwise use it as voices list
      else
        GetRecordVoiceTypes(LinksTo(ElementByName(Alias, 'VTCK - Voice Types')), lstVoice);
    end;

    // stop processing other aliases
    Break;
  end;
end;

//============================================================================
// get voice types from conditions
procedure GetConditionsVoiceTypes(Conditions: IInterface; lstVoice: TStringList);
var
  Condition, Elem, Quest: IInterface;
  ConditionFunction: string;
  lstVoiceCondition: TStringList;
  i, j, Alias: integer;
  bFactionCondition, bGetIsID: Boolean;
begin
  AddDebug('GetConditionsVoiceTypes ' + FullPath(Conditions));

  lstVoiceCondition := TStringList.Create; lstVoiceCondition.Duplicates := dupIgnore; lstVoiceCondition.Sorted := True;

  // check all condition functions beforehand
  // GetIsID has priority over everything else, if it is found ignore other functions
  for i := 0 to Pred(ElementCount(Conditions)) do
    if GetElementEditValues(ElementByIndex(Conditions, i), 'CTDA\Function') = 'GetIsID' then begin
      bGetIsID := True;
      Break;
    end;

  for i := 0 to Pred(ElementCount(Conditions)) do begin
    Condition := ElementByIndex(Conditions, i);
    ConditionFunction := GetElementEditValues(Condition, 'CTDA\Function');
    // NPC
    if ConditionFunction = 'GetIsID' then begin
      InfoCONDITION := ConditionFunction;
      Elem := LinksTo(ElementByPath(Condition, 'CTDA\Referenceable Object'));
      GetRecordVoiceTypes(Elem, lstVoiceCondition);
    end else
    // skip other functions
    if not bGetIsID then 
    // Voice type of FLST list of voice types
    if ConditionFunction = 'GetIsVoiceType' then begin
      InfoCONDITION := ConditionFunction;
      Elem := LinksTo(ElementByPath(Condition, 'CTDA\Voice Type'));
      GetRecordVoiceTypes(Elem, lstVoiceCondition);
    end
    // Quest alias
    else if ConditionFunction = 'GetIsAliasRef' then begin
      Alias := GetElementNativeValues(Condition, 'CTDA\Alias');
      Elem := ContainingMainRecord(Conditions);
      if Signature(Elem) = 'INFO' then begin
        Elem := LinksTo(ElementByName(Elem, 'Topic'));
        Quest := LinksTo(ElementByName(Elem, 'QNAM - Quest'));
      end
      else if Signature(Elem) = 'QUST' then
        Quest := Elem;
      GetAliasVoiceTypes(Quest, Alias, lstVoiceCondition);
    end else
    if ConditionFunction = 'GetInFaction' then begin
      // looks like GetInFaction is used only once by CK when exporting dialogues, i.e. 000D9513
      if not bFactionCondition then begin
        bFactionCondition := True;
        Elem := LinksTo(ElementByPath(Condition, 'CTDA\Faction'));
        GetRecordVoiceTypes(Elem, lstVoiceCondition);
      end;
    end
    else if ConditionFunction = 'GetIsClass' then begin
      Elem := LinksTo(ElementByPath(Condition, 'CTDA\Class'));
      GetRecordVoiceTypes(Elem, lstVoiceCondition);
    end;
    
    if lstVoiceCondition.Count = 0 then
      Continue;
    
    // not sure about how to combine voice types from several conditions, for now just OR
    lstVoice.AddStrings(lstVoiceCondition);
    
    {// if condition is ORed then combine voice lists
    if GetElementNativeValues(Condition, 'CTDA\Type') and 1 > 0 then
      lstVoice.AddStrings(lstVoiceCondition)
    // otherwise intersect (AND)
    else begin
      for j := Pred(lstVoice.Count) downto 0 do
        if lstVoiceCondition.IndexOf(lstVoice[j]) = -1 then
          lstVoice.Delete(j);
    end;
    }
    lstVoiceCondition.Clear;
  end;
  
  lstVoiceCondition.Free;
end;

//============================================================================
// get a list of voice types for INFO record
procedure InfoVoiceTypes(Info: IInterface; lstVoice: TStringList);
var
  Elem, Dialogue: IInterface;
  Conditions: IInterface;
  Scene, Actions, Action: IInterface;
  i, j, Alias: integer;
  bAliasFound: Boolean;
begin
  if not Assigned(lstVoice) then
    Exit;

  lstVoice.Clear;
  
  // check Speaker
  Elem := ElementByName(Info, 'ANAM - Speaker');
  if Assigned(Elem) then begin
    Elem := LinksTo(Elem);
    GetRecordVoiceTypes(Elem, lstVoice);
    InfoSPEAKER := Name(Elem);
    Exit;
  end;
  
  // check Conditions
  if ElementExists(Info, 'Conditions') then
    GetConditionsVoiceTypes(ElementByName(Info, 'Conditions'), lstVoice);
  
  // check Scene aliases if above has no result
  if lstVoice.Count <> 0 then
    Exit;

  bAliasFound := False;
  Dialogue := LinksTo(ElementByName(Info, 'Topic'));

  if GetElementEditValues(Dialogue, 'DATA\Category') = 'Scene' then begin
    for i := Pred(ReferencedByCount(Dialogue)) downto 0 do begin
      Scene := ReferencedByIndex(Dialogue, i);
      if Signature(Scene) <> 'SCEN' then Continue;
      
      AddDebug('Searching for alias and dialog in scene ' + Name(Scene));
      Actions := ElementByName(Scene, 'Actions');
      for j := 0 to Pred(ElementCount(Actions)) do begin
        Action := ElementByIndex(Actions, j);
        if (GetElementEditValues(Action, 'ANAM - Type') = 'Dialogue') and
           (GetElementNativeValues(Action, 'DATA - Topic') = GetLoadOrderFormID(Dialogue))
        then begin
          Alias := GetElementNativeValues(Action, 'ALID - Actor ID');
          GetAliasVoiceTypes(LinksTo(ElementByName(Scene, 'PNAM - Quest')), Alias, lstVoice);
          bAliasFound := True;
          Break;
        end;
      end;
      if bAliasFound then
        Break;
    end;
  end;

  // if still can't determine voices, then use all of them
  if lstVoice.Count = 0 then begin
    AddMessage('Warning: No voice types found for ' + Name(Info));
    //GetRecordVoiceTypes(RecordByFormID(FileByIndex(0), $0003B4A5, False), lstVoice); // DefaultNPCVoiceTypes [FLST:0003B4A5]
  end;
  
end;

//============================================================================
// adds child INFOs of DIAL record to list
procedure AddInfosFromDial(Dialogue: IInterface; lst: TStringList);
var
  i: integer;
  InfoGroup, Info: IInterface;
begin
  InfoGroup := ChildGroup(Dialogue);
  for i := 0 to Pred(ElementCount(InfoGroup)) do begin
    Info := ElementByIndex(InfoGroup, i);
    lst.AddObject(IntToHex(GetLoadOrderFormID(Info), 8), Info);
  end;
end;

//============================================================================
procedure ExportQuest(Quest: IInterface);
var
  i, j, r, v: Integer;
  lstDial, lstInfo, lstVoice: TStringList;
  Dialogue, DialogueOvr, Info: IInterface;
  Responses, Response: IInterface;
  ResponseNumber: Integer;
  VoiceFileName, VoiceFilePath, PluginName: string;
  Scene, Actions, Action, PlayerDialogue: IInterface;
  s, a: Integer;
  IsPlayerDialogue, IsNPCDialogue: Boolean;
  StartAlias, TopicID: Integer;
  ActionType: string;
begin
  // Prepare lists
  lstDial := TStringList.Create; lstDial.Sorted := True; lstDial.Duplicates := dupIgnore;
  lstInfo := TStringList.Create; lstInfo.Sorted := True; lstInfo.Duplicates := dupIgnore;
  lstVoice := TStringList.Create; lstVoice.Sorted := True; lstVoice.Duplicates := dupIgnore;
  PluginName := GetFileName(GetFile(Quest));

  // 1) Gather all DIALs referencing this quest
  for i := 0 to Pred(ReferencedByCount(Quest)) do
    if Signature(ReferencedByIndex(Quest, i)) = 'DIAL' then
      lstDial.AddObject(
        IntToHex(GetLoadOrderFormID(ReferencedByIndex(Quest, i)), 8),
        ReferencedByIndex(Quest, i)
      );

  // 2) Process each dialogue
  for i := 0 to Pred(lstDial.Count) do begin
    Dialogue := ObjectToElement(lstDial.Objects[i]);
    if GetIsDeleted(Dialogue) then Continue;

    // Build list of all INFO overrides + master
    lstInfo.Clear;
    for j := Pred(OverrideCount(Dialogue)) downto 0 do
      AddInfosFromDial(OverrideByIndex(Dialogue, j), lstInfo);
    AddInfosFromDial(Dialogue, lstInfo);

    // Process each INFO
    for j := 0 to Pred(lstInfo.Count) do begin
      Info := ObjectToElement(lstInfo.Objects[j]);
      if GetIsDeleted(Info) then Continue;

      // Skip if no responses
      Responses := ElementByName(Info, 'Responses');
      if not Assigned(Responses) then Continue;

      // Reset flags & capture topic FormID
      IsPlayerDialogue := False;
      IsNPCDialogue    := False;
      StartAlias       := -1;
      TopicID          := GetLoadOrderFormID(Dialogue);

      // If this is a Scene‐type topic, scan its SCENs
      if GetElementEditValues(Dialogue, 'DATA\Category') = 'Scene' then begin
        for s := 0 to Pred(ReferencedByCount(Dialogue)) do begin
          Scene := ReferencedByIndex(Dialogue, s);
          if Signature(Scene) <> 'SCEN' then Continue;

          Actions := ElementByName(Scene, 'Actions');

          // Find the Start Scene alias
          for a := 0 to Pred(ElementCount(Actions)) do begin
            Action := ElementByIndex(Actions, a);
            if GetElementEditValues(Action, 'ANAM - Type') = 'Start Scene' then begin
              StartAlias := GetElementNativeValues(Action, 'ALID - Actor ID');
              Break;
            end;
          end;

          // Now inspect each action for player vs NPC dialogue
          for a := 0 to Pred(ElementCount(Actions)) do begin
            Action := ElementByIndex(Actions, a);
            ActionType := GetElementEditValues(Action, 'ANAM - Type');

            if ActionType = 'Player Dialogue' then begin
              PlayerDialogue := ElementByName(Action, 'Player Dialogue');

              // Player tags
              if ((GetElementNativeValues(PlayerDialogue, 'PTOP') = TopicID) or
                  (GetElementNativeValues(PlayerDialogue, 'NTOP') = TopicID) or
                  (GetElementNativeValues(PlayerDialogue, 'NETO') = TopicID) or
                  (GetElementNativeValues(PlayerDialogue, 'QTOP') = TopicID))
              then begin
                IsPlayerDialogue := True;
                Break;
              end;

              // NPC tags
              if ((GetElementNativeValues(PlayerDialogue, 'NPOT') = TopicID) or
                  (GetElementNativeValues(PlayerDialogue, 'NNGT') = TopicID) or
                  (GetElementNativeValues(PlayerDialogue, 'NNUT') = TopicID) or
                  (GetElementNativeValues(PlayerDialogue, 'NQUT') = TopicID) or
                  (GetElementNativeValues(PlayerDialogue, 'NNGS') = TopicID) or
                  (GetElementNativeValues(PlayerDialogue, 'NNUS') = TopicID) or
                  (GetElementNativeValues(PlayerDialogue, 'NQUS') = TopicID))
              then begin
                IsNPCDialogue := True;
                Break;
              end;
            end;
          end;

          if IsPlayerDialogue or IsNPCDialogue then
            Break;
        end;
      end;

      // Build the voice list
      lstVoice.Clear;
      if IsPlayerDialogue then begin
        lstVoice.Add('PlayerVoiceFemale01');
        lstVoice.Add('PlayerVoiceMale01');
      end
      else if IsNPCDialogue and (StartAlias >= 0) then begin
        GetAliasVoiceTypes(Quest, StartAlias, lstVoice);
      end
      else begin
        InfoVoiceTypes(Info, lstVoice);
      end;

      // Export each response × each voice
      for r := 0 to Pred(ElementCount(Responses)) do begin
        Response := ElementByIndex(Responses, r);
        ResponseNumber := GetElementNativeValues(Response, 'TRDT\Response number') + 1;

        if lstVoice.Count > 0 then begin
          for v := 0 to Pred(lstVoice.Count) do begin
            VoiceFileName := InfoFileName(
              GetLoadOrderFormID(Info),
              ResponseNumber
            );
            VoiceFilePath := Format(
              'Data\Sound\Voice\%s\%s\',
              [GetFileName(Quest), lstVoice[v]]
            );
            slExport.Add(Format(
              '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s',
              [
                EscapeCSVField(GetFileName(Quest)),
                EscapeCSVField(Name(Quest)),
                EscapeCSVField(InfoNPCID),
                EscapeCSVField(InfoSPEAKER),
                EscapeCSVField(InfoRACEID),
                EscapeCSVField(lstVoice[v]),
                EscapeCSVField(InfoCONDITION),
                EscapeCSVField(EditorID(LinksTo(ElementByName(Dialogue, 'BNAM - Branch')))),
                EscapeCSVField(GetElementEditValues(Dialogue, 'DATA\Category')),
                EscapeCSVField(GetElementEditValues(Dialogue, 'SNAM')),
                EscapeCSVField(GetElementEditValues(Dialogue, 'DATA\Subtype')),
                EscapeCSVField(Trim(EditorID(Info) + ' [INFO:' + IntToHex(FormID(Info), 8) + ']')),
                EscapeCSVField(IntToStr(ResponseNumber)),
                EscapeCSVField(VoiceFileName),
                EscapeCSVField(VoiceFilePath + VoiceFileName + '.fuz'),
                EscapeCSVField(GetElementEditValues(Dialogue, 'FULL')),
                EscapeCSVField(GetElementEditValues(Info, 'RNAM - Prompt')),
                EscapeCSVField(GetElementEditValues(Response, 'NAM1 - Response Text')),
                EscapeCSVField(GetElementEditValues(Response, 'TRDT\Emotion Type') + ' ' +
                  GetElementEditValues(Response, 'TRDT\Emotion Value')),
                EscapeCSVField(GetElementEditValues(Response, 'NAM2 - Script Notes'))
              ]
            ));
          end;
        end
        else begin
          // No voice found fallback
          slExport.Add(Format(
            '%s,%s,%s,%s,%s,"no voice",%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s',
            [
              EscapeCSVField(GetFileName(Quest)),
              EscapeCSVField(Name(Quest)),
              EscapeCSVField(InfoNPCID),
              EscapeCSVField(InfoSPEAKER),
              EscapeCSVField(InfoRACEID),
              EscapeCSVField(InfoCONDITION),
              EscapeCSVField(EditorID(LinksTo(ElementByName(Dialogue, 'BNAM - Branch')))),
              EscapeCSVField(GetElementEditValues(Dialogue, 'DATA\Category')),
              EscapeCSVField(GetElementEditValues(Dialogue, 'SNAM')),
              EscapeCSVField(GetElementEditValues(Dialogue, 'DATA\Subtype')),
              EscapeCSVField(Trim(EditorID(Info) + ' [INFO:' + IntToHex(FormID(Info), 8) + ']')),
              EscapeCSVField(IntToStr(ResponseNumber)),
              EscapeCSVField(GetElementEditValues(Dialogue, 'FULL')),
              EscapeCSVField(GetElementEditValues(Info, 'RNAM - Prompt')),
              EscapeCSVField(GetElementEditValues(Response, 'NAM1 - Response Text')),
              EscapeCSVField(GetElementEditValues(Response, 'TRDT\Emotion Type') + ' ' +
                GetElementEditValues(Response, 'TRDT\Emotion Value')),
              EscapeCSVField(GetElementEditValues(Response, 'NAM2 - Script Notes'))
            ]
          ));
        end;
      end;
    end; // for each INFO

  end; // for each DIAL

  // Clean up
  lstDial.Free;
  lstInfo.Free;
  lstVoice.Free;

  ExportFileName := PluginName + '_' + '_dialogueExport.csv';
end;

//============================================================================
function Process(e: IInterface): integer;
var
  s: string;
  sl: TStringList;
begin
  if Signature(e) <> 'QUST' then
    Exit;

  AddMessage('=== EXPORTING: ' + Name(e));
  e := MasterOrSelf(e);
  
  if ContainerStates(GetFile(e)) and (1 shl csRefsBuild) = 0 then begin
    AddMessage('Skipping quest, references are not built for file ' + GetFileName(e));
    AddMessage('Use Right click \ Other \ Build Reference Info menu and try again.');
    Exit;
  end;

  ExportQuest(e);
end;

//============================================================================
function Finalize: integer;
var
  dlgSave: TSaveDialog;
begin
  if slExport.Count <> 0 then begin
    AddMessage('Saving ' + ExportFileName);
    slExport.Insert(0, 'PLUGIN,QUEST,NPCID,SPEAKER,RACEID,VOICE TYPE,CONDITION,BRANCH,CATEGORY,TYPE,SUBTYPE,TOPIC,RESPONSE INDEX,FILENAME,FULLPATH,TOPIC TEXT,PROMPT,RESPONSE TEXT,EMOTION,SCRIPT NOTES');
    slExport.SaveToFile(ExportFileName);
  end;
  slExport.Free;
  lstRecursion.Free;
end;

end.
