#!/bin/sh
:<<!
Author: Qianqian Peng
Data: 2021-4-11
Description: this shell script is built to use MetaMap to annotate PubTator file.
!
#22663310
set -- `getopt i:o: $*`

getopt_rc=$?

if [ "$getopt_rc" -ne "0" ]; then
  pgm=`basename $0`
  echo "$pgm $getopt_rc Process failed during getopt attempt - illegal parameters"
  echo "Usage is: $0 [-i] indir [-o] outdir"
  exit 10
fi

while [ $# -gt 0 ]; do
  case $1 in
    -i)
       shift
       indir=$1
#       echo "-i is $indir"
       shift
       ;;
    -o)
       shift
       outdir=$1
#       echo "-o is $outdir"
       shift
       ;;
    --)
       shift
       echo "Usage is: $0 [-i] indir [-o] outdir"
       shift
       break
  esac
done

if [ ! -d $outdir ]
then
  mkdir $outdir
  echo 'mkdir '$outdir
fi

prefix='out_'
for infile in $(ls $indir)
do
  infile2=$indir/temp.PubTator
  sed 's/[ ][ ]*/ /g' $indir/$infile > $infile2
  outfile=$outdir/$prefix$infile
  metamap20 -G $infile2 $outfile
  rm -rf $infile2
done
