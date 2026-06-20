$ErrorActionPreference = 'Stop'

$Distro = if ($args.Count -gt 0) { $args[0] } else { 'Ubuntu' }
$ProjectDir = '/mnt/d/Pytnon/olc_overlap_framework_v0_1_0/olc_overlap_framework'
$Command = "cd $ProjectDir && source .venv/bin/activate && exec bash -i"

wsl -d $Distro -- bash -lc $Command
