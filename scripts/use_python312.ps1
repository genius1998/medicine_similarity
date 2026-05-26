$pythonRoot = "C:\Users\com\AppData\Local\Programs\Python\Python312"
$pythonExe = Join-Path $pythonRoot "python.exe"

if (-not (Test-Path $pythonExe)) {
    throw "Python 3.12 interpreter not found at $pythonExe"
}

$pythonScripts = Join-Path $pythonRoot "Scripts"
$currentPath = @($env:Path -split ";") | Where-Object { $_ }
$filteredPath = $currentPath | Where-Object {
    $_ -ne $pythonRoot -and $_ -ne $pythonScripts
}
$env:Path = (($pythonRoot, $pythonScripts) + $filteredPath) -join ";"
$env:PY_PYTHON = "3.12"

Write-Host "Python 3.12 configured for this PowerShell session." -ForegroundColor Green
& $pythonExe --version
