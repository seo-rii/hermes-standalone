"""Exercise kernel-format battery facts using disposable sysfs fixtures."""

from pathlib import Path
import sys
import tempfile
import unittest

from standalone.power import linux_battery


@unittest.skipUnless(sys.platform.startswith("linux"), "Linux power_supply ABI")
class BatteryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="hermes-power-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def supply(self, name, **values):
        directory = self.root / name
        directory.mkdir()
        for key, value in values.items():
            (directory / key).write_text(str(value) + "\n", encoding="ascii")
        return directory

    def test_charge_and_energy_units_are_kept_separate(self):
        battery = self.supply("BAT0", type="Battery", status="Discharging", energy_now=24000000,
                              energy_full=48000000, power_now=6000000, charge_now=999)
        status = linux_battery(self.root)
        self.assertEqual(status, (50.0, 14400, False))
        (battery / "energy_full").unlink()
        (battery / "charge_now").write_text("2000000\n")
        (battery / "charge_full").write_text("5000000\n")
        (battery / "current_now").write_text("-500000\n")
        self.assertEqual(linux_battery(self.root), (40.0, 14400, False))

    def test_capacity_and_kernel_time_override_derived_values(self):
        battery = self.supply("BAT0", type="Battery", capacity=72, status="Discharging",
                              time_to_empty_now=1234)
        self.assertEqual(linux_battery(self.root), (72.0, 1234, False))
        (battery / "status").write_text("Charging\n")
        self.assertEqual(linux_battery(self.root), (72.0, -2, True))

    def test_missing_and_unknown_power_states_are_not_fabricated(self):
        self.assertIsNone(linux_battery(self.root))
        battery = self.supply("BAT0", type="Battery", present=0, capacity=50)
        self.assertIsNone(linux_battery(self.root))
        (battery / "present").write_text("1\n")
        self.assertEqual(linux_battery(self.root), (50.0, -1, None))
        self.supply("AC", type="Mains", online=1)
        self.assertEqual(linux_battery(self.root), (50.0, -2, True))


if __name__ == "__main__":
    unittest.main()
