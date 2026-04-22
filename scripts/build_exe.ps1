param(
  [ValidateSet("onedir", "onefile")]
  [string]$Mode = "onedir"
)

$ErrorActionPreference = "Stop"

$py = "python"
if (Test-Path ".venv\\Scripts\\python.exe") { $py = ".venv\\Scripts\\python.exe" }

& $py -m pip install -r requirements.txt | Out-Null

$specDir = "build\\pyinstaller"
New-Item -ItemType Directory -Force -Path $specDir | Out-Null

if ($Mode -eq "onefile") {
  cmd.exe /c "taskkill /F /IM VaultHUD.exe >nul 2>nul"
  & $py -m PyInstaller --noconfirm --clean --onefile --name VaultHUD --specpath $specDir --workpath $specDir vault_hud.py
  Write-Host "Built: dist\VaultHUD.exe"
} else {
  cmd.exe /c "taskkill /F /IM VaultHUD.exe >nul 2>nul"
  & $py -m PyInstaller --noconfirm --clean --onedir --name VaultHUD --specpath $specDir --workpath $specDir vault_hud.py
  Write-Host "Built: dist\\VaultHUD\\VaultHUD.exe"
}

