# Modified from Alphafold2 codes

"""Library to run HHblits from Python."""

import glob
import os
import subprocess
from typing import Any, Mapping, Optional, Sequence
from gate.tool import utils

_HHBLITS_DEFAULT_P = 20
_HHBLITS_DEFAULT_Z = 500


class HHBlits:
  """Python wrapper of the HHblits binary."""

  def __init__(self,
               *,
               binary_path: str,
               databases: Sequence[str],
               n_cpu: int = 4,
               n_iter: int = 3,
               e_value: float = 0.001,
               maxseq: int = 1_000_000,
               realign_max: int = 100_000,
               maxfilt: int = 100_000,
               min_prefilter_hits: int = 1000,
               all_seqs: bool = False,
               alt: Optional[int] = None,
               p: int = _HHBLITS_DEFAULT_P,
               z: int = _HHBLITS_DEFAULT_Z):
    """Initializes the Python HHblits wrapper.
    Args:
      binary_path: The path to the HHblits executable.
      databases: A sequence of HHblits database paths. This should be the
        common prefix for the database files (i.e. up to but not including
        _hhm.ffindex etc.)
      n_cpu: The number of CPUs to give HHblits.
      n_iter: The number of HHblits iterations.
      e_value: The E-value, see HHblits docs for more details.
      maxseq: The maximum number of rows in an input alignment. Note that this
        parameter is only supported in HHBlits version 3.1 and higher.
      realign_max: Max number of HMM-HMM hits to realign. HHblits default: 500.
      maxfilt: Max number of hits allowed to pass the 2nd prefilter.
        HHblits default: 20000.
      min_prefilter_hits: Min number of hits to pass prefilter.
        HHblits default: 100.
      all_seqs: Return all sequences in the MSA / Do not filter the result MSA.
        HHblits default: False.
      alt: Show up to this many alternative alignments.
      p: Minimum Prob for a hit to be included in the output hhr file.
        HHblits default: 20.
      z: Hard cap on number of hits reported in the hhr file.
        HHblits default: 500. NB: The relevant HHblits flag is -Z not -z.
    Raises:
      RuntimeError: If HHblits binary not found within the path.
    """
    self.binary_path = binary_path
    self.databases = databases

    for database_path in self.databases:
      print(f"Using database: {database_path}")
      if not glob.glob(database_path + '_*'):
        raise ValueError(f'Could not find HHBlits database {database_path}')

    self.n_cpu = n_cpu
    self.n_iter = n_iter
    self.e_value = e_value
    self.maxseq = maxseq
    self.realign_max = realign_max
    self.maxfilt = maxfilt
    self.min_prefilter_hits = min_prefilter_hits
    self.all_seqs = all_seqs
    self.alt = alt
    self.p = p
    self.z = z

  def query(self, input_fasta_path: str, output_a3m_path: str) -> Mapping[str, Any]:
      """Queries the database using HHblits."""

      targetname = open(input_fasta_path).readlines()[0].rstrip('\n').lstrip('>')

      db_cmd = []
      for db_path in self.databases:
          db_cmd.append('-d')
          db_cmd.append(db_path)
      cmd = [
          self.binary_path,
          '-i', input_fasta_path,
          '-cpu', str(self.n_cpu),
          '-oa3m', output_a3m_path,
          '-o', f'{output_a3m_path}.log',
          '-n', str(self.n_iter),
          '-e', str(self.e_value),
          '-maxseq', str(self.maxseq),
          '-realign_max', str(self.realign_max),
          '-maxfilt', str(self.maxfilt),
          '-min_prefilter_hits', str(self.min_prefilter_hits)]
      if self.all_seqs:
          cmd += ['-all']
      if self.alt:
          cmd += ['-alt', str(self.alt)]
      if self.p != _HHBLITS_DEFAULT_P:
          cmd += ['-p', str(self.p)]
      if self.z != _HHBLITS_DEFAULT_Z:
          cmd += ['-Z', str(self.z)]
      cmd += db_cmd

      print(cmd)
      process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

      stdout, stderr = process.communicate()
      retcode = process.wait()

      if retcode:
          # Logs have a 15k character limit, so log HHblits error line by line.
          raise RuntimeError('HHblits failed\nstdout:\n%s\n\nstderr:\n%s\n' % (
              stdout.decode('utf-8'), stderr[:500_000].decode('utf-8')))

      raw_output = dict(
          a3m=output_a3m_path,
          output=stdout,
          stderr=stderr,
          n_iter=self.n_iter,
          e_value=self.e_value)
      return raw_output
