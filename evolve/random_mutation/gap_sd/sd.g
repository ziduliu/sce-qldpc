# StructureDescription of every library group, one task of NT (task index TI, both set by the caller).
Read("/u/zdliu/deploy/rnd_v5_20261015/gap_sd/pairs.g");
out := OutputTextFile(Concatenation("/u/zdliu/deploy/rnd_v5_20261015/gap_sd/sd_", String(TI), ".tsv"), false);
SetPrintFormattingStatus(out, false);
for i in [1..Length(PAIRS)] do
  if (i - 1) mod NT = TI then
    p := PAIRS[i];
    AppendTo(out, p[1], "\t", p[2], "\t", StructureDescription(SmallGroup(p[1], p[2])), "\n");
  fi;
od;
CloseStream(out);
QUIT;
