#!/usr/bin/env python3
"""Analyze the frequency of precision landing (PL) messages in an ArduPilot BIN log
and count how many separate landing attempts the log contains.

A landing attempt starts when the vehicle switches into a landing mode (see
--land-modes) and ends when it switches to a non-landing mode, disarms, or the
log ends.

Usage:
    python3 scripts/frequency.py bin_files/precision_landing.BIN
    python3 scripts/frequency.py bin_files/precision_landing.BIN --type PL --bin-seconds 5
    python3 scripts/frequency.py bin_files/precision_landing.BIN --land-modes LAND
"""

import argparse
import bisect
import statistics
import sys

from pymavlink import mavutil

# AP_Logger LogEvent ids used to delimit landing attempts
EV_DISARMED = 11
EV_LAND_COMPLETE = 18
EV_NOT_LANDED = 28

DEFAULT_LAND_MODES = "LAND,RTL,SMART_RTL,AUTO_RTL"


def _mode_name(msg):
    """Best-effort flight mode name from a MODE dataflash message."""
    mode = getattr(msg, "Mode", None)
    if isinstance(mode, str):
        return mode.upper()
    num = getattr(msg, "ModeNum", None)
    if num is None:
        num = mode
    try:
        # numeric fallback assumes copter mode numbers
        return mavutil.mode_mapping_acm.get(int(num), str(num))
    except (TypeError, ValueError):
        return str(num)


def collect_log_data(log_path: str, msg_type: str):
    """Read msg_type, MODE and EV messages from the log in a single pass."""
    mlog = mavutil.mavlink_connection(log_path)
    timestamps_us = []
    acquired_flags = []
    mode_changes = []  # (TimeUS, mode name)
    events = []        # (TimeUS, event id)
    while True:
        msg = mlog.recv_match(type=[msg_type, "MODE", "EV"])
        if msg is None:
            break
        mtype = msg.get_type()
        if mtype == msg_type:
            timestamps_us.append(msg.TimeUS)
            acquired_flags.append(getattr(msg, "TAcq", None))
        if mtype == "MODE":
            mode_changes.append((msg.TimeUS, _mode_name(msg)))
        elif mtype == "EV":
            events.append((msg.TimeUS, msg.Id))
    return timestamps_us, acquired_flags, mode_changes, events


def detect_landing_attempts(mode_changes, events, land_modes):
    """Group MODE/EV messages into landing attempts.

    Returns a list of dicts with start_us, end_us (None = log end), modes,
    touchdown_us and end_reason.
    """
    timeline = sorted(
        [(t, "mode", m) for t, m in mode_changes]
        + [(t, "event", ev) for t, ev in events],
        key=lambda item: item[0],
    )
    attempts = []
    current = None

    def close(end_us, reason):
        nonlocal current
        current["end_us"] = end_us
        current["end_reason"] = reason
        attempts.append(current)
        current = None

    for t_us, kind, value in timeline:
        if kind == "mode":
            if value in land_modes:
                if current is None:
                    current = {"start_us": t_us, "modes": [value],
                               "touchdown_us": None}
                elif current["modes"][-1] != value:
                    # e.g. RTL -> LAND mid-descent: same attempt, note the path
                    current["modes"].append(value)
            elif current is not None:
                close(t_us, f"mode change to {value}")
        elif current is not None:
            if value == EV_LAND_COMPLETE and current["touchdown_us"] is None:
                current["touchdown_us"] = t_us
            elif value == EV_NOT_LANDED:
                current["touchdown_us"] = None
            elif value == EV_DISARMED:
                close(t_us, "disarmed")
    if current is not None:
        close(None, "log end")
    return attempts


def print_landing_attempts(attempts, timestamps_us, acquired_flags,
                           land_modes, log_end_us, msg_type):
    print(f"=== Landing attempts (modes: {', '.join(sorted(land_modes))}) ===")
    if not attempts:
        print("No landing attempts detected (no MODE change into a landing mode).")
        return

    print(f"Attempts:          {len(attempts)}")
    for i, attempt in enumerate(attempts, 1):
        start_us = attempt["start_us"]
        end_us = attempt["end_us"] if attempt["end_us"] is not None else log_end_us
        t0, t1 = start_us / 1e6, end_us / 1e6
        print(f"  #{i}: {'>'.join(attempt['modes']):<10} "
              f"t={t0:8.1f} -> {t1:8.1f} s ({t1 - t0:6.1f} s)  "
              f"ended: {attempt['end_reason']}")
        if attempt["touchdown_us"] is not None:
            print(f"      touchdown (LAND_COMPLETE) at t={attempt['touchdown_us'] / 1e6:.1f} s")

        i0 = bisect.bisect_left(timestamps_us, start_us)
        i1 = bisect.bisect_right(timestamps_us, end_us)
        count = i1 - i0
        if count:
            duration = t1 - t0
            rate = count / duration if duration > 0 else 0.0
            known = [f for f in acquired_flags[i0:i1] if f is not None]
            acq_txt = ""
            if known:
                acquired = sum(1 for f in known if f)
                acq_txt = f", target acquired {acquired / len(known) * 100:.1f}%"
            print(f"      {msg_type} msgs: {count} @ {rate:.1f} Hz{acq_txt}")
        else:
            print(f"      {msg_type} msgs: none during this attempt")


def print_summary(timestamps_us, acquired_flags, msg_type: str, bin_seconds: float):
    count = len(timestamps_us)
    if count < 2:
        print(f"Found {count} {msg_type} message(s) - not enough to compute frequency.")
        return

    t = [us / 1e6 for us in timestamps_us]  # seconds since boot
    duration = t[-1] - t[0]
    intervals = [b - a for a, b in zip(t, t[1:])]

    print(f"=== {msg_type} message frequency summary ===")
    print(f"Messages:          {count}")
    print(f"First message:     {t[0]:.3f} s (time since boot)")
    print(f"Last message:      {t[-1]:.3f} s")
    print(f"Span:              {duration:.3f} s")
    print(f"Average rate:      {(count - 1) / duration:.2f} Hz")
    print()

    print("--- Inter-message intervals ---")
    print(f"Mean:              {statistics.mean(intervals) * 1000:.2f} ms")
    print(f"Median:            {statistics.median(intervals) * 1000:.2f} ms")
    print(f"Min:               {min(intervals) * 1000:.2f} ms")
    print(f"Max:               {max(intervals) * 1000:.2f} ms")
    print(f"Std dev:           {statistics.stdev(intervals) * 1000:.2f} ms")
    print()

    gap_threshold = 3 * statistics.median(intervals)
    gaps = [(t[i], iv) for i, iv in enumerate(intervals) if iv > gap_threshold]
    print(f"--- Gaps (> 3x median interval, {gap_threshold * 1000:.1f} ms) ---")
    if gaps:
        for start, iv in sorted(gaps, key=lambda g: -g[1])[:10]:
            print(f"  {iv * 1000:8.1f} ms gap starting at t={start:.3f} s")
        if len(gaps) > 10:
            print(f"  ... and {len(gaps) - 10} more")
    else:
        print("  none")
    print()

    if any(f is not None for f in acquired_flags):
        acquired = sum(1 for f in acquired_flags if f)
        print("--- Target acquisition (TAcq) ---")
        print(f"Target acquired:   {acquired} ({acquired / count * 100:.1f}%)")
        print(f"Not acquired:      {count - acquired} ({(count - acquired) / count * 100:.1f}%)")
        print()

    print(f"--- Rate over time ({bin_seconds:g} s bins) ---")
    bin_start = t[0]
    bin_count = 0
    for ts in t:
        while ts >= bin_start + bin_seconds:
            print(f"  t={bin_start:8.1f} - {bin_start + bin_seconds:8.1f} s: "
                  f"{bin_count / bin_seconds:6.1f} Hz  ({bin_count} msgs)")
            bin_start += bin_seconds
            bin_count = 0
        bin_count += 1
    if bin_count:
        remaining = t[-1] - bin_start
        rate = bin_count / remaining if remaining > 0 else 0.0
        print(f"  t={bin_start:8.1f} - {t[-1]:8.1f} s: {rate:6.1f} Hz  ({bin_count} msgs)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", help="path to the .BIN dataflash log")
    parser.add_argument("--type", default="PL",
                        help="message type to analyze (default: PL)")
    parser.add_argument("--bin-seconds", type=float, default=10.0,
                        help="bin size in seconds for the rate-over-time table (default: 10)")
    parser.add_argument("--land-modes", default=DEFAULT_LAND_MODES,
                        help="comma-separated modes that count as landing attempts "
                             f"(default: {DEFAULT_LAND_MODES})")
    args = parser.parse_args()

    timestamps_us, acquired_flags, mode_changes, events = collect_log_data(
        args.log, args.type)
    land_modes = {m.strip().upper() for m in args.land_modes.split(",") if m.strip()}

    if timestamps_us:
        print_summary(timestamps_us, acquired_flags, args.type, args.bin_seconds)
        print()
    else:
        print(f"No {args.type} messages found in {args.log}")
        print()

    attempts = detect_landing_attempts(mode_changes, events, land_modes)
    log_end_us = max(
        timestamps_us[-1] if timestamps_us else 0,
        mode_changes[-1][0] if mode_changes else 0,
        events[-1][0] if events else 0,
    )
    print_landing_attempts(attempts, timestamps_us, acquired_flags,
                           land_modes, log_end_us, args.type)

    if not timestamps_us:
        sys.exit(1)


if __name__ == "__main__":
    main()
