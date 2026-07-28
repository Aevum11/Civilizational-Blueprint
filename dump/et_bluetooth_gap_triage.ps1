# ============================================================================
# ET BLUETOOTH DESCRIPTOR-GAP TRIAGE — Windows PowerShell (run as Administrator)
# Exception Theory LLC — method: the Three Tools; mechanisms: named plainly.
#
# THE GAP (Descriptor Gap Principle): headphones fine on phone, bad on PC at
# EVERY range -> propagation eliminated; the differing descriptors are the PC's
# stack + local RF environment. This script closes every software-reachable one.
# HONESTY: no software raises TX power. These are the real levers:
#   1. Profile: Windows falling back to Handsfree (8 kHz mono) instead of A2DP
#   2. Power management strangling the BT radio and its USB hub
#   3. WiFi camping on 2.4 GHz (Bluetooth's band) instead of 5 GHz
#   4. USB 3.0 broadband noise in 2.4 GHz (Intel-documented) — hardware advice
# ============================================================================

if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
    ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
  Write-Host "Run this as Administrator." -ForegroundColor Red; exit 1
}
Write-Host "`n=== ET BT TRIAGE — Identification / Gap / Subsumption ===" -ForegroundColor Cyan

# ── §1 IDENTIFY the radio and its descriptors ──────────────────────────────
$bt = Get-PnpDevice -Class Bluetooth -Status OK | Where-Object {
        $_.FriendlyName -match "Radio|Adapter|Wireless Bluetooth|Intel|Realtek|MediaTek|Qualcomm" }
if ($bt) { $bt | ForEach-Object { Write-Host ("Radio: " + $_.FriendlyName) } }
else     { Write-Host "No BT radio enumerated — check adapter presence." -ForegroundColor Yellow }

# ── §2 THE PROFILE GAP (the #1 audio-quality culprit) ──────────────────────
# Windows enables Handsfree Telephony; any app touching the mic drags the link
# to HFP (8 kHz mono). Disabling the HFP endpoint pins high-quality A2DP.
$hfp = Get-PnpDevice | Where-Object { $_.FriendlyName -match "Hands-?Free" -and $_.Status -eq "OK" }
if ($hfp) {
  foreach ($d in $hfp) {
    Write-Host ("Disabling Handsfree endpoint: " + $d.FriendlyName) -ForegroundColor Green
    Disable-PnpDevice -InstanceId $d.InstanceId -Confirm:$false -ErrorAction SilentlyContinue
  }
  Write-Host "  (Re-enable in Device Manager if you ever need the headset mic.)"
} else { Write-Host "No active Handsfree endpoint found (good, or headphones not connected)." }

# ── §3 POWER MANAGEMENT: stop the OS strangling the radio ──────────────────
foreach ($d in $bt) {
  $key = "HKLM:\SYSTEM\CurrentControlSet\Enum\$($d.InstanceId)\Device Parameters"
  try {
    New-ItemProperty -Path $key -Name "SelectiveSuspendEnabled" -Value 0 `
      -PropertyType Binary -Force -ErrorAction Stop | Out-Null
    Write-Host ("Selective suspend OFF for " + $d.FriendlyName) -ForegroundColor Green
  } catch { Write-Host ("Could not set suspend key for " + $d.FriendlyName) -ForegroundColor Yellow }
}
# Global USB selective suspend off (AC and DC)
powercfg /SETACVALUEINDEX SCHEME_CURRENT 2a737441-1930-4402-8d77-b2bebba308a3 48e6b7a6-50f5-4782-a5d4-53bb8f07e226 0 | Out-Null
powercfg /SETDCVALUEINDEX SCHEME_CURRENT 2a737441-1930-4402-8d77-b2bebba308a3 48e6b7a6-50f5-4782-a5d4-53bb8f07e226 0 | Out-Null
powercfg /SETACTIVE SCHEME_CURRENT | Out-Null
Write-Host "USB selective suspend disabled (AC + battery)." -ForegroundColor Green

# ── §4 BAND COEXISTENCE: is WiFi sitting on Bluetooth's 2.4 GHz? ───────────
$wifi = netsh wlan show interfaces
$band = ($wifi | Select-String "Band|Channel")
Write-Host "`nWiFi status:"; $band | ForEach-Object { Write-Host ("  " + $_.Line.Trim()) }
if ($wifi -match "2\.4 GHz" -or ($wifi -match "Channel\s*:\s*([0-9]+)" -and [int]$Matches[1] -le 14)) {
  Write-Host "GAP FOUND: WiFi is on 2.4 GHz — the SAME band Bluetooth hops in." -ForegroundColor Yellow
  Write-Host "  Fix: connect to your router's 5 GHz SSID (or set band preference to" 
  Write-Host "  5 GHz in the WiFi adapter's Advanced properties). Vacating the band"
  Write-Host "  is the single biggest coexistence win."
} else { Write-Host "WiFi is off 2.4 GHz (or disconnected) — band clear for BT." -ForegroundColor Green }

# ── §5 STACK RESTART: renegotiate the link cleanly ─────────────────────────
Restart-Service bthserv -Force -ErrorAction SilentlyContinue
Write-Host "`nBluetooth support service restarted. Re-pair the headphones once now:"
Write-Host "  Remove device -> pair fresh (forces clean A2DP negotiation)."

# ── §6 THE HARDWARE-LEVEL DESCRIPTOR (no script can reach it) ──────────────
Write-Host "`n=== USB 3.0 INTERFERENCE (Intel-documented; hardware-level) ===" -ForegroundColor Cyan
Write-Host "USB 3.0 ports and cables radiate broadband noise across 2.4-2.5 GHz —"
Write-Host "directly on top of Bluetooth. If your BT is a USB dongle, or the internal"
Write-Host "antenna routes near the port cluster, this is why the phone (clean RF"
Write-Host "environment, dedicated antenna) wins at identical range."
Write-Host "  Fix A: move a BT dongle to a USB 2.0 port, or onto a short USB 2.0"
Write-Host "         extension cable away from all USB 3 ports/devices."
Write-Host "  Fix B: keep external SSDs / USB 3 hubs on the far side of the case."
Write-Host "`nDone. Reboot, re-pair, test. If audio is now full-quality and stable,"
Write-Host "the gap was profile/power/band. If dropouts persist ONLY near USB 3"
Write-Host "activity, the gap is Fix A — two dollars of extension cable." 
