@echo off
chcp 65001 >nul
title Rebuild hypergraph
echo Rebuilding hypergraph on the cluster ...
ssh cluster "bash /path/to/cluster_build.sh"
echo.
echo Done. Press any key to close.
pause >nul
