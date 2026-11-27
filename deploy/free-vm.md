# Free Ubuntu VM for the ArbSense collector

Use an Oracle Cloud Free Tier tenancy in its home region. In Compute → Instances choose an Always Free eligible Ubuntu 24.04 Arm image and VM.Standard.A1.Flex with 1 OCPU, 6 GB RAM and a 50 GB standard boot volume. Check the tenancy's remaining free compute and **total** storage allowance, including existing volumes. Keep the free account type. If capacity is unavailable, try another availability domain in the same home region. [Oracle free resource limits](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm)

Use a public subnet with an internet gateway, an ephemeral public IP and SSH ingress on TCP 22 restricted to your laptop's public IPv4 address (/32). The collector uses outbound HTTPS, secure WebSockets and NTP. In the creation form save the generated private SSH key on your laptop, or upload your existing public key. Only the public key belongs on the VM.

## Transfer the tested checkout

On Windows, from the local ArbSense checkout, set the IP and private key path to the ones from your instance. These commands transfer committed project files and Git history. Ignored local databases and environment files stay on the laptop.

```powershell
$vmIp = Read-Host 'VM public IP'
$vmKey = Read-Host 'Path to your private SSH key'
$vmBundle = Join-Path $env:TEMP 'arbsense-vm.bundle'
git bundle create $vmBundle main
scp -i $vmKey $vmBundle "ubuntu@${vmIp}:arbsense-vm.bundle"
ssh -i $vmKey "ubuntu@${vmIp}"
```

On the VM:

```bash
sudo apt-get update && sudo apt-get install -y git
git clone -b main ~/arbsense-vm.bundle ~/arbsense
cd ~/arbsense
bash deploy/setup-ubuntu-collector.sh
git status --short --branch
```

The installer checks that the collector loads and prints the active pairs and available disk. It leaves the collector stopped. The earlier laptop estimate was 4.36 GB of database storage over five days. Allow additional space for the OS, WAL and exported results.

## Verify the VM before collection

When ready, run a ten-minute smoke capture in its own database:

```bash
cd ~/arbsense
.venv/bin/python scripts/run_live.py 600 data/vm-smoke.db
.venv/bin/python scripts/propose_latency_threshold.py --db data/vm-smoke.db
```

Check that all feeds actually supplied data from this region, the clock synced, and the warm-up completed. Once the timing and fee assumptions have been recorded and committed, start the bounded five-day capture from the clean checkout using a separate database:

```bash
cd ~/arbsense
sudo systemd-run --unit=arbsense-five-day --uid="$(id -un)" --working-directory="$PWD" "$PWD/.venv/bin/python" -u scripts/run_live.py 432000 "$PWD/data/vm-five-day.db"
journalctl -u arbsense-five-day -f
```

The transient service continues after SSH disconnects and ends when the collector finishes. A VM reboot interrupts it. Preserve the database and record any interruption before resuming a capture. For early shutdown use `sudo systemctl stop arbsense-five-day`.
