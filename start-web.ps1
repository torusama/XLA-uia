param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8765
)

$ErrorActionPreference = 'Stop'
$repoPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$previousUtf8 = $env:PYTHONUTF8
$previousPythonPath = $env:PYTHONPATH

function Invoke-Python {
    & $python @args
    if ($LASTEXITCODE -ne 0) {
        throw "Python exited with code $LASTEXITCODE. See the error above."
    }
}

Push-Location -LiteralPath $PSScriptRoot
try {
    $env:PYTHONUTF8 = '1'
    $sourcePath = Join-Path $PSScriptRoot 'src'
    $env:PYTHONPATH = if ($previousPythonPath) {
        "$sourcePath$([IO.Path]::PathSeparator)$previousPythonPath"
    } else {
        $sourcePath
    }

    if ($env:VIRTUAL_ENV -and (Test-Path -LiteralPath "$env:VIRTUAL_ENV/Scripts/python.exe")) {
        $python = Join-Path $env:VIRTUAL_ENV 'Scripts/python.exe'
    } elseif (Test-Path -LiteralPath $repoPython) {
        $python = $repoPython
    } else {
        $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
        $launcherArgs = @('-3')
        if (-not $launcher) {
            $launcher = Get-Command python.exe -ErrorAction SilentlyContinue
            $launcherArgs = @()
        }
        if (-not $launcher) { throw 'Install Python 3.10 or newer and run this script again.' }
        & $launcher.Source @launcherArgs -c 'import sys; sys.exit(sys.version_info < (3, 10))'
        if ($LASTEXITCODE -ne 0) { throw 'Python 3.10 or newer is required.' }
        & $launcher.Source @launcherArgs -m venv --system-site-packages .venv
        if ($LASTEXITCODE -ne 0) { throw 'Could not create .venv.' }
        $python = $repoPython
    }

    Write-Host "Python: $python"
    Write-Host 'Checking dependencies. Compatible installations are reused.'
    Invoke-Python -c 'import sys; sys.exit(sys.version_info < (3, 10))'
    Invoke-Python -m pip --disable-pip-version-check install --quiet -r requirements.txt
    & $python -c "from importlib.util import find_spec; raise SystemExit(find_spec('efficientvit') is None)"
    if ($LASTEXITCODE -ne 0) {
        $efficientvit = 'https://github.com/mit-han-lab/efficientvit/archive/de7d7733cc0329f391b33f1f459271562ec27bd5.zip'
        Invoke-Python -m pip --disable-pip-version-check install --quiet --no-deps $efficientvit
    }

    Invoke-Python scripts/prepare_demo.py

    Write-Host "Open http://127.0.0.1:$Port when the server is ready. Press Ctrl+C to stop."
    Invoke-Python -u -m demo.backend.server --port $Port
}
finally {
    Pop-Location
    $env:PYTHONUTF8 = $previousUtf8
    $env:PYTHONPATH = $previousPythonPath
}
