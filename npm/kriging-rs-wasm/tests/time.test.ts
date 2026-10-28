import { describe, test, expect } from "vitest";

import { datesFromTimes, timesFromDates } from "../dist/index.js";

describe("time helpers", () => {
  const jan1 = new Date(Date.UTC(2024, 0, 1));
  const feb1 = new Date(Date.UTC(2024, 1, 1));
  const feb1Plus5Min = new Date(Date.UTC(2024, 1, 1, 0, 5, 0));

  test("timesFromDates defaults to days from Unix epoch", () => {
    const t = timesFromDates([jan1, feb1]);
    expect(t).toBeInstanceOf(Float64Array);
    expect(t.length).toBe(2);
    expect(t[1] - t[0]).toBe(31);
  });

  test("timesFromDates supports every TimeUnit", () => {
    expect(
      timesFromDates([jan1, feb1], "ms")[1] -
        timesFromDates([jan1, feb1], "ms")[0]
    ).toBe(31 * 86_400_000);
    expect(
      timesFromDates([jan1, feb1], "s")[1] -
        timesFromDates([jan1, feb1], "s")[0]
    ).toBe(31 * 86_400);
    expect(
      timesFromDates([jan1, feb1Plus5Min], "minutes")[1] -
        timesFromDates([jan1, feb1Plus5Min], "minutes")[0]
    ).toBe(31 * 24 * 60 + 5);
    expect(
      timesFromDates([jan1, feb1Plus5Min], "hours")[1] -
        timesFromDates([jan1, feb1Plus5Min], "hours")[0]
    ).toBeCloseTo(31 * 24 + 5 / 60, 10);
  });

  test("datesFromTimes is the inverse of timesFromDates", () => {
    const originals = [jan1, feb1, feb1Plus5Min];
    for (const unit of ["ms", "s", "minutes", "hours", "days"] as const) {
      const round = datesFromTimes(timesFromDates(originals, unit), unit);
      expect(round.length).toBe(originals.length);
      for (let i = 0; i < originals.length; i++) {
        expect(round[i].getTime()).toBe(originals[i].getTime());
      }
    }
  });

  test("timesFromDates honors a non-default epoch", () => {
    const epoch = new Date(Date.UTC(2024, 0, 1));
    const t = timesFromDates([jan1, feb1], "days", epoch);
    expect(t[0]).toBe(0);
    expect(t[1]).toBe(31);
  });

  test("datesFromTimes accepts plain arrays and Float64Arrays", () => {
    const fromArray = datesFromTimes([0, 1], "days");
    const fromTyped = datesFromTimes(new Float64Array([0, 1]), "days");
    expect(fromArray.map((d) => d.getTime())).toEqual(
      fromTyped.map((d) => d.getTime())
    );
  });
});
