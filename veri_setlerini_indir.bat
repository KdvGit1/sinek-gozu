@echo off
rem ==========================================================================
rem  Drone veri setlerini indirir (data\downloads) ve data\raw\<set> altina acar.
rem
rem  - Kaldigi yerden devam: yarim dosya .part olarak durur, curl -C - ile
rem    surdurulur. Pencereyi kapatip tekrar calistirmak yeterli.
rem  - Tekrar deneme: baglanti koparsa WAIT_SEC saniye bekleyip devam eder.
rem    Ilerleme oldugu surece denemeye devam eder; ust uste MAX_TRY denemede
rem    hic ilerleme olmazsa o seti atlayip digerine gecer.
rem  - Dogrulama: boyut (hepsi) + MD5 (kaynak verdiyse). Acilan set
rem    .extracted_* isaretiyle kaydedilir, tekrar calistirinca atlanir.
rem
rem  Kullanim:
rem    veri_setlerini_indir.bat                    hepsi (~50 GB indirme)
rem    veri_setlerini_indir.bat hit_uav birdsai    sadece secilenler
rem  Setler: wisard rgbtdroneperson hit_uav birdsai visdrone
rem ==========================================================================
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

rem ---- Ayarlar -------------------------------------------------------------
set "MAX_TRY=20"
set "WAIT_SEC=30"
rem 1 yapilirsa acilan zip silinir (WiSARD icin ~43 GB yer kazandirir)
set "DELETE_ZIP=0"
rem --------------------------------------------------------------------------

set "CURL=%SystemRoot%\System32\curl.exe"
set "TAR=%SystemRoot%\System32\tar.exe"
set "DL=%CD%\data\downloads"
set "RAW=%CD%\data\raw"
set "LOG=%DL%\indirme.log"
set "GD=https://drive.usercontent.google.com/download?export=download&confirm=t&id="
set "ALL=wisard rgbtdroneperson hit_uav birdsai visdrone"
set "RC=1"

if not exist "%CURL%" (echo HATA: %CURL% bulunamadi. Windows 10 1803 veya ustu gerekli.& goto :end)
if not exist "%TAR%"  (echo HATA: %TAR% bulunamadi. Windows 10 1803 veya ustu gerekli.& goto :end)
if not exist "%DL%" mkdir "%DL%"
if not exist "%RAW%" mkdir "%RAW%"

set "SETS=%*"
if "%SETS%"=="" set "SETS=%ALL%"
for %%S in (%SETS%) do (
    set "OK="
    for %%K in (%ALL%) do if /i "%%S"=="%%K" set "OK=1"
    if not defined OK (
        echo Bilinmeyen set: %%S
        echo Gecerli setler: %ALL%
        goto :end
    )
)

echo Indirme klasoru: %DL%
echo Cikti klasoru  : %RAW%
echo Secilen setler : %SETS%
echo.
set "FAILED="
for %%S in (%SETS%) do (
    echo ==========================================================================
    echo  %%S
    echo ==========================================================================
    call :set_%%S
    if errorlevel 1 (
        set "FAILED=!FAILED! %%S"
        echo [!date! !time!] BASARISIZ %%S>>"%LOG%"
    ) else (
        echo [!date! !time!] TAMAM %%S>>"%LOG%"
    )
    echo.
)

echo ==========================================================================
if defined FAILED (
    echo Tamamlanamayan setler:%FAILED%
    echo Betigi tekrar calistirin; inen kisimlar korunur, kaldigi yerden devam eder.
    set "RC=1"
) else (
    echo Hepsi tamam. Sonraki adim: ADIMLAR.md - 1.2 ^(inspect^)
    set "RC=0"
)
goto :end


rem ==========================================================================
rem  Setler
rem  :get  <dosya> <url> <bayt> <md5 veya -> <hedef klasor> <strip>
rem ==========================================================================

:set_wisard
rem MIT - Google Drive, 43.5 GB (Visual Only + Thermal Only + Multi-Modal)
call :get "WiSARDv1.zip" "%GD%1PKjGCqUszHH1nMbXUBTwPSDqRabAt_ht" 43526830098 - "%RAW%\wisard" 0
exit /b

:set_rgbtdroneperson
rem CC BY 4.0 - Google Drive. Zip icinde VOC xml var; COCO json'lar ayri dosyalar.
call :get "RGBTDronePerson.zip" "%GD%18zm2CaJS73a6eUHXUQ6_cjjbrurSkm9C" 1573829771 - "%RAW%\rgbtdroneperson" 0 || exit /b 1
set "ANN=%RAW%\rgbtdroneperson\annotations"
if not exist "%ANN%" mkdir "%ANN%"
call :fetch "%ANN%\train_thermal.json"     "%GD%1P5FNdig1_IEq9xGfL3zAzMxQf-Jeg7_2" 10265189 - || exit /b 1
call :fetch "%ANN%\val_thermal.json"       "%GD%1H_hHqoqaVPBa3LBTEgveZLxE-eovaOmE" 2636638 -  || exit /b 1
call :fetch "%ANN%\sub_train_thermal.json" "%GD%1Cxp6Hau20jRQowEAU7ST5wwI_kRUQq65" 2105751 -  || exit /b 1
call :fetch "%ANN%\sub_train_visible.json" "%GD%1aAlgSY8tWA4RDIi88l83mFtrdvj0KAGa" 2084741 -  || exit /b 1
exit /b 0

:set_hit_uav
rem Zenodo 10.5281/zenodo.7633134 (GitHub v1.2.1 arsivi, goruntuler dahil)
call :get "HIT-UAV-v1.2.1.zip" "https://zenodo.org/api/records/7633134/files/suojiashun/HIT-UAV-Infrared-Thermal-Dataset-v1.2.1.zip/content" 814751578 bb6c5a33659b7507eae83458729c6109 "%RAW%\hit_uav" 1
exit /b

:set_birdsai
rem CDLA-Permissive 1.0 - LILA (Google Cloud). Sentetik veri (39.5 GB) indirilmez.
call :get "conservation_drones_train_real.zip" "https://storage.googleapis.com/public-datasets-lila/conservationdrones/v01/conservation_drones_train_real.zip" 2271707323 802d0baf11dbe6f38f9840d88591ba0e "%RAW%\birdsai" 0 || exit /b 1
call :get "conservation_drones_test_real.zip" "https://storage.googleapis.com/public-datasets-lila/conservationdrones/v01/conservation_drones_test_real.zip" 1761788520 1d5d67338e2c402b78dc2f454f502ba9 "%RAW%\birdsai" 0
exit /b

:set_visdrone
rem CC BY-NC-SA 3.0 - SADECE TEST. Resmi setin Ultralytics yansisi (val kumesi).
call :get "VisDrone2019-DET-val.zip" "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip" 81638851 f44ac10dd2bd69a4325509071c6ed449 "%RAW%\visdrone" 0
exit /b


rem ==========================================================================
rem  :get <dosya> <url> <bayt> <md5> <hedef> <strip>  -> indir + ac
rem ==========================================================================
:get
set "G_MARK=%~5\.extracted_%~n1"
if exist "%G_MARK%" (
    echo   [atlandi] %~1 daha once acilmis.
    exit /b 0
)
call :fetch "%DL%\%~1" "%~2" %3 %4 || exit /b 1
call :extract "%DL%\%~1" "%~5" %6 || exit /b 1
exit /b 0


rem ==========================================================================
rem  :fetch <cikti yolu> <url> <bayt> <md5 veya ->
rem ==========================================================================
:fetch
set "F_OUT=%~1"
set "F_NAME=%~nx1"
set "F_PART=%~1.part"
set "F_URL=%~2"
set "F_SIZE=%~3"
set "F_MD5=%~4"

if exist "%F_OUT%" (
    call :size_of "%F_OUT%" CUR
    if "!CUR!"=="%F_SIZE%" (
        echo   [atlandi] %F_NAME% zaten indirilmis.
        exit /b 0
    )
    echo   [uyari] %F_NAME% boyutu hatali ^(!CUR! / %F_SIZE%^), yeniden indirilecek.
    del /q "%F_OUT%"
)

call :pad %F_SIZE% PSIZE
set /a F_TRY=0
set "PREV=-1"

:fetch_loop
call :size_of "%F_PART%" CUR
if "!CUR!"=="%F_SIZE%" goto :fetch_verify
call :pad !CUR! PCUR
if "!PCUR!" GTR "!PSIZE!" (
    echo   [uyari] %F_NAME%.part beklenenden buyuk, bozuk. Silinip bastan indirilecek.
    del /q "%F_PART%"
    set "CUR=0"
)
rem Kucuk bir .part HTML ile basliyorsa (eski surumden kalan kota sayfasi) silinir
if "!PCUR!" LSS "000000001048576" if exist "%F_PART%" (
    findstr /r /b /i /c:"[<].doctype" /c:"[<]html" "%F_PART%" >nul 2>&1 && (
        echo   [uyari] %F_NAME%.part dosya degil web sayfasi iceriyor, silindi.
        del /q "%F_PART%"
        set "CUR=0"
    )
)
rem Ilerleme olduysa deneme sayaci sifirlanir
if not "!CUR!"=="!PREV!" set /a F_TRY=0
set "PREV=!CUR!"
set /a F_TRY+=1
if !F_TRY! GTR %MAX_TRY% (
    echo   HATA: %F_NAME% ust uste %MAX_TRY% denemede ilerlemedi.
    echo         Link: !F_URL!
    echo         Tarayicidan indirip "%F_OUT%" olarak koyarsaniz betik onu kullanir.
    exit /b 1
)

if "!CUR!"=="0" (
    echo   Indiriliyor: %F_NAME%  ^(%F_SIZE% bayt, deneme !F_TRY!/%MAX_TRY%^)
) else (
    echo   Devam ediliyor: %F_NAME%  ^(!CUR! / %F_SIZE% bayt, deneme !F_TRY!/%MAX_TRY%^)
)

rem Sunucu dosya yerine HTML donduyse (Google Drive kota/hata sayfasi) .part'a
rem eklenmesin diye once 1 baytlik istekle icerik tipine bakilir.
set "CT="
for /f "delims=" %%c in ('""%CURL%" -s -L -r 0-0 -o nul --connect-timeout 30 --max-time 120 -w "%%{content_type}" "!F_URL!""') do set "CT=%%c"
if not defined CT (
    echo   [tekrar] Sunucuya ulasilamadi. %WAIT_SEC% sn sonra tekrar denenecek.
    call :wait
    goto :fetch_loop
)
if /i not "!CT:text/html=!"=="!CT!" (
    echo   [tekrar] Sunucu dosya yerine web sayfasi dondurdu ^(Google Drive kotasi dolmus olabilir^).
    echo       %WAIT_SEC% sn sonra tekrar denenecek.
    call :wait
    goto :fetch_loop
)

rem Asil indirme. On kontrol gecse bile asil istek kota sayfasi alabiliyor (WiSARD'da oldu);
rem bu yuzden yanitin icerik tipi de okunur ve HTML geldiyse .part eski boyutuna geri kesilir.
rem Ilerleme cubugu stderr'e yazdigi icin ekranda gorunmeye devam eder.
set "PRE=!CUR!"
set "RES="
for /f "delims=" %%r in ('""%CURL%" -L --fail -C - --retry 3 --retry-delay 10 --connect-timeout 30 --speed-limit 10240 --speed-time 120 -o "!F_PART!" -w "%%{exitcode} %%{content_type}" "!F_URL!""') do set "RES=%%r"
set "CRC=99"
set "RCT="
if defined RES for /f "tokens=1,*" %%a in ("!RES!") do (
    set "CRC=%%a"
    set "RCT=%%b"
)
if defined RCT if /i not "!RCT:text/html=!"=="!RCT!" (
    echo   [tekrar] Sunucu dosya yerine web sayfasi dondurdu ^(Google Drive kotasi dolmus olabilir^).
    if "!PRE!"=="0" (
        del /q "%F_PART%" 2>nul
    ) else (
        powershell -NoProfile -Command "$f=[IO.File]::Open($env:F_PART,'Open');$f.SetLength([long]$env:PRE);$f.Close()"
    )
    echo       .part eski boyutuna geri alindi. %WAIT_SEC% sn sonra tekrar denenecek.
    call :wait
    goto :fetch_loop
)
if not "!CRC!"=="0" (
    echo   [tekrar] Baglanti hatasi ^(curl kodu !CRC!^). %WAIT_SEC% sn sonra kaldigi yerden devam edilecek.
    call :wait
)
goto :fetch_loop

:fetch_verify
if not "%F_MD5%"=="-" (
    echo   MD5 kontrol ediliyor: %F_NAME% ...
    set "GOT="
    for /f "skip=1 delims=" %%h in ('certutil -hashfile "%F_PART%" MD5') do if not defined GOT set "GOT=%%h"
    set "GOT=!GOT: =!"
    if /i not "!GOT!"=="%F_MD5%" (
        echo   HATA: MD5 uyusmadi ^(!GOT!^). Dosya silindi, betigi tekrar calistirin.
        del /q "%F_PART%"
        exit /b 1
    )
    echo   MD5 dogru.
)
move /y "%F_PART%" "%F_OUT%" >nul
echo   [tamam] %F_NAME%
echo [%date% %time%] indirildi %F_NAME% %F_SIZE%>>"%LOG%"
exit /b 0


rem ==========================================================================
rem  :extract <zip> <hedef klasor> <strip-components>
rem ==========================================================================
:extract
set "X_MARK=%~2\.extracted_%~n1"
if not exist "%~2" mkdir "%~2"
set "X_OPT="
if not "%~3"=="0" set "X_OPT=--strip-components %~3"
echo   Aciliyor: %~nx1 -^> %~2
"%TAR%" -xf "%~1" -C "%~2" %X_OPT%
if errorlevel 1 (
    echo   HATA: %~nx1 acilamadi. Zip bozuk olabilir; silip betigi tekrar calistirin.
    exit /b 1
)
echo %date% %time%>"%X_MARK%"
echo   [tamam] acildi.
if "%DELETE_ZIP%"=="1" del /q "%~1"
exit /b 0


rem ==========================================================================
rem  Yardimcilar
rem ==========================================================================
:size_of
rem z niteleyicisi buyuk dosyalarda da (4 GB ustu) dogru boyutu metin olarak verir
set "%~2=0"
if exist "%~1" set "%~2=%~z1"
exit /b 0

:pad
rem Sayilari 15 haneye doldurur; cmd 32-bit oldugu icin buyuk boyutlar
rem metin olarak karsilastirilir
set "P_TMP=000000000000000%~1"
set "%~2=!P_TMP:~-15!"
exit /b 0

:wait
timeout /t %WAIT_SEC% /nobreak >nul 2>&1 || ping -n %WAIT_SEC% 127.0.0.1 >nul
exit /b 0

:end
echo.
rem Cift tiklamayla acildiysa pencere hemen kapanmasin
echo %cmdcmdline% | find /i "%~0" >nul && pause
endlocal & exit /b %RC%
