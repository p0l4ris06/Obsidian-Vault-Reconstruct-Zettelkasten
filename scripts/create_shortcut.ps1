param(
  [string]$ExePath = "",
  [ValidateSet("Desktop", "StartMenu", "Both")]
  [string]$Location = "Desktop",
  [string]$ShortcutName = "VaultHUD"
)

$ErrorActionPreference = "Stop"

function Resolve-ExePath {
  param([string]$PathArg)

  if ($PathArg -and (Test-Path -LiteralPath $PathArg)) {
    return (Resolve-Path -LiteralPath $PathArg).Path
  }

  $default = Join-Path (Get-Location) "dist\VaultHUD_onedir\VaultHUD_onedir.exe"
  if (Test-Path -LiteralPath $default) {
    return (Resolve-Path -LiteralPath $default).Path
  }

  throw "Could not find VaultHUD_onedir.exe. Build it first with: .\build_exe.ps1 -Mode onedir"
}

$exe = Resolve-ExePath -PathArg $ExePath
$startIn = Split-Path -Parent $exe

$desktopDir = [Environment]::GetFolderPath("Desktop")
$startMenuDir = Join-Path ([Environment]::GetFolderPath("StartMenu")) "Programs"

$targets = @()
if ($Location -eq "Desktop" -or $Location -eq "Both") { $targets += (Join-Path $desktopDir "$ShortcutName.lnk") }
if ($Location -eq "StartMenu" -or $Location -eq "Both") { $targets += (Join-Path $startMenuDir "$ShortcutName.lnk") }

$shell = New-Object -ComObject WScript.Shell
foreach ($lnk in $targets) {
  $sc = $shell.CreateShortcut($lnk)
  $sc.TargetPath = $exe
  $sc.WorkingDirectory = $startIn
  $sc.Arguments = ""
  $sc.WindowStyle = 1
  $sc.Description = "VaultHUD (Zettelkasten command center)"
  $sc.Save()
  Write-Host "Created shortcut: $lnk"
}

