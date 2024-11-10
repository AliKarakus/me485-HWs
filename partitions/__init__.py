from partitions.metis import METISPartition


def get_partition(msh, out, npart):
    return METISPartition(msh, out, npart)
