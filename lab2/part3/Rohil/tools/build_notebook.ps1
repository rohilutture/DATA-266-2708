# Converts src/<Name>.py (percent format) into notebooks/<Name>.ipynb. Usage: powershell tools/build_notebook.ps1 [-Name p3_agents]
param([string]$Name = "p2_finetune")
$root = Split-Path -Parent $PSScriptRoot
$lines = Get-Content -Encoding UTF8 (Join-Path $root "src\$Name.py")
$cells = @(); $cur = $null
function Flush($cur) {
  if ($null -eq $cur) { return $null }
  $src = @($cur.lines)
  while ($src.Count -gt 0 -and $src[-1].Trim() -eq "") { $src = $src[0..($src.Count - 2)] }
  if ($src.Count -eq 0) { return $null }
  if ($cur.type -eq "markdown") { $src = $src | ForEach-Object { if ($_ -match '^# ?(.*)$') { $Matches[1] } else { $_ } } }
  $arr = @(); for ($i = 0; $i -lt $src.Count; $i++) { $arr += ($src[$i] + $(if ($i -lt $src.Count - 1) { "`n" } else { "" })) }
  $c = [ordered]@{ cell_type = $cur.type; metadata = @{}; source = $arr }
  if ($cur.type -eq "code") { $c.execution_count = $null; $c.outputs = @() }
  return $c
}
foreach ($l in $lines) {
  if ($l -match '^# %%( \[markdown\])?\s*$') {
    $c = Flush $cur; if ($c) { $cells += , $c }
    $cur = @{ type = $(if ($l -match 'markdown') { "markdown" } else { "code" }); lines = @() }
  } elseif ($cur) { $cur.lines += $l }
}
$c = Flush $cur; if ($c) { $cells += , $c }
$nb = [ordered]@{ cells = $cells; metadata = @{ kernelspec = @{ display_name = "Python 3"; language = "python"; name = "python3" }; language_info = @{ name = "python" } }; nbformat = 4; nbformat_minor = 5 }
$outDir = Join-Path $root "notebooks"; New-Item -ItemType Directory -Force $outDir | Out-Null
$json = $nb | ConvertTo-Json -Depth 10
[System.IO.File]::WriteAllText((Join-Path $outDir "$Name.ipynb"), $json, (New-Object System.Text.UTF8Encoding $false))
"cells: $($cells.Count)"
