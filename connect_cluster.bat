@echo off
chcp 65001 >nul
title MMGraph-Med cluster
echo Connecting to cluster ...
ssh -t cluster "bash /path/to/cluster_session.sh"
echo.
echo Disconnected. Press any key to close.
pause >nul
