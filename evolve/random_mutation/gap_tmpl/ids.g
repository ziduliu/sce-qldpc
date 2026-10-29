# IdGroup of every template presentation of rnd_mutator.py: metacyclic <a,b | a^m, b^r, b a b^-1 = a^s>
# and dicyclic <x,y | x^(2n), y^2 = x^n, y x y^-1 = x^-1>. Output: tag q p1 p2 p3 id (tab separated).
Read("/u/zdliu/deploy/rnd_v5_20261015/gap_tmpl/tmpl.g");
out := OutputTextFile("/u/zdliu/deploy/rnd_v5_20261015/gap_tmpl/tmpl_ids.tsv", false);
SetPrintFormattingStatus(out, false);
for t in TMPL do
  F := FreeGroup("a", "b"); a := F.1; b := F.2;
  if t[1] = "mc" then
    G := F / [a^t[3], b^t[4], b*a*b^-1*a^(-t[5])];
  else
    G := F / [a^(2*t[3]), b^2*a^(-t[3]), b*a*b^-1*a];
  fi;
  if Size(G) <> t[2] then Error("size mismatch ", t); fi;
  AppendTo(out, t[1], "\t", t[2], "\t", t[3], "\t", t[4], "\t", t[5], "\t", IdGroup(G)[2], "\n");
od;
CloseStream(out);
QUIT;
