param(
  [ValidateSet("onedir", "onefile")]
  [string]$Mode = "onedir"
)

$ErrorActionPreference = "Stop"

# Back-compat wrapper: older docs/scripts referenced `build_hud.ps1`.
& "$PSScriptRoot\\build_exe.ps1" -Mode $Mode
