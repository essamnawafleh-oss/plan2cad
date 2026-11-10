; Plan2CAD installer (Inno Setup 6)
#define AppName "Plan2CAD"
#define AppVersion "1.0"
#define AppPublisher "Essam Nawafleh - عصام النوافلة"
#define AppExe "Plan2CAD.exe"

[Setup]
AppId={{6F1C2B7E-8A3D-4E2F-9B51-2C7D9E4A1F30}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion} - Essam Nawafleh
AppPublisher={#AppPublisher}
AppCopyright=© Essam Nawafleh - عصام النوافلة
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
OutputDir=output
OutputBaseFilename=Plan2CAD_Setup
SetupIconFile=p2c\assets\icon.ico
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName} - Essam Nawafleh
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
ChangesAssociations=yes

[Languages]
Name: "arabic"; MessagesFile: "compiler:Languages\Arabic.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "dist\Plan2CAD\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Registry]
Root: HKA; Subkey: "Software\Classes\.p2c"; ValueType: string; ValueName: ""; ValueData: "Plan2CAD.Project"; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\Plan2CAD.Project"; ValueType: string; ValueName: ""; ValueData: "Plan2CAD Project"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\Plan2CAD.Project\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\{#AppExe},0"
Root: HKA; Subkey: "Software\Classes\Plan2CAD.Project\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent
