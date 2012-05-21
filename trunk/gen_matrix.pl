#!/usr/bin/perl

$size = 750;

my $rows = 5 + $size;
my $cols = 11 + $size;

print "$rows\n";
print "$cols\n";

# x1 < 40
print "1\n0\n0\n0\n0\n";

print "1\n";
for (my $i = 0; $i < $size; $i++) {
  print "0\n";
}
print "0\n0\n0\n0\n40\n";

# x1 < 40 + $z

for (my $z = 1; $z <= $size; $z++) {
  print "1\n0\n0\n0\n0\n";
  print "0\n";
  for (my $i = 1; $i <= $size; $i++) {
    print "0\n" if ($i != $z);
	print "1\n" if ($i == $z);
  }
  print "0\n0\n0\n0\n";
  print 40 + $z . "\n";
}

# x2 + x3 < 35
print "0\n1\n1\n0\n0\n";

print "0\n";
for (my $i = 0; $i < $size; $i++) {
  print "0\n";
}
print "1\n0\n0\n0\n35\n";

# x3 + x4 < 45
print "0\n0\n1\n1\n0\n";

print "0\n";
for (my $i = 0; $i < $size; $i++) {
  print "0\n";
}
print "0\n1\n0\n0\n45\n";

# x2 + x3 + x5 < 28
print "0\n1\n1\n0\n1\n";

print "0\n";
for (my $i = 0; $i < $size; $i++) {
  print "0\n";
}
print "0\n0\n1\n0\n28\n";

# 5x1 + 2x2 + 3x3 + 4x4 - 2 x5 < Z
print "-5\n-2\n-3\n-4\n2\n";

print "0\n";
for (my $i = 0; $i < $size; $i++) {
  print "0\n";
}
print "0\n0\n0\n1\n0";