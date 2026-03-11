@echo off
cd /d Q:\finance-analyzer
echo [%date% %time%] PF-SilverORB is legacy. Delegating to scripts\win\silver-monitor.bat...
call scripts\win\silver-monitor.bat
