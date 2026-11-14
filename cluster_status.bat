@echo off
chcp 65001 >nul
title Cluster status
ssh cluster "bash /path/to/cluster_status.sh"
pause >nul
