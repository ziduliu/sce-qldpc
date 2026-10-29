# Export the GAP small groups library as permutation generators (JSON lines) for the
# full-library random-mutation arm: every group of order 5..299 except 256, plus a fixed
# uniform sample of 300 of the 56,092 groups of order 256.
LoadPackage("smallgrp");
out := OutputTextFile("/u/zdliu/lp/jobs/smallgrp_lib_20261011/smallgrp_lib.jsonl", false);
SetPrintFormattingStatus(out, false);
Reset(GlobalMersenneTwister, 20261011);
n256 := NumberSmallGroups(256);
ids256 := Set(Shuffle([1..n256]){[1..300]});
count := 0;
for q in [5..299] do
  if q = 256 then ids := ids256; else ids := [1..NumberSmallGroups(q)]; fi;
  for i in ids do
    G := SmallGroup(q, i);
    P := Image(IsomorphismPermGroup(G));
    Q := Image(SmallerDegreePermutationRepresentation(P));
    d := LargestMovedPoint(Q);
    if d = 0 then d := 1; fi;
    gens := List(GeneratorsOfGroup(Q), g -> List(ListPerm(g, d), x -> x - 1));
    AppendTo(out, "{\"q\":", q, ",\"id\":", i, ",\"deg\":", d, ",\"abelian\":", IsAbelian(G), ",\"gens\":", gens, "}\n");
    count := count + 1;
  od;
  if q mod 20 = 0 then Print("order ", q, " done, ", count, " groups so far\n"); fi;
od;
CloseStream(out);
Print("DONE ", count, " groups\n");
QUIT;
