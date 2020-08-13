import unittest
from join_metasamples_metadata import JoinMetasamplesMetadata


class JoinMetadataMetasamplesTests(unittest.TestCase):

    def setUp(self):
        """ We test with factors (metagenes) and expression matrix from Factors/Mini_AOCS and
        metadata from Data/Mini_AOCS, both of which are committed.  Note
        that the metadata rows (patients/samples) are deliberately NOT in the same order
        as in the columns of the expression matrix.   The code has to deal with this."""

        self.jmsmd = JoinMetasamplesMetadata('Mini_AOCS', 'Mini_AOCS')

    def test_init(self):
        assert self.jmsmd is not None

    def test_read_metadata_and_x_dfs(self):
        meta_df, x_df = self.jmsmd.read_metadata_and_x_dfs()
        print(meta_df.columns)
        print(x_df.columns)
        # It is crucial that these dataframes are aligned with respect to patients!
        assert list(meta_df.index.values) == list(x_df.columns)

    def test_read_factors_df(self):
        meta_df, x_df = self.jmsmd.read_metadata_and_x_dfs()
        factor_df = self.jmsmd.read_factors_df(x_df, 'ICA', 5)
        # print(factor_df)
        # print(meta_df)
        assert list(meta_df.index.values) == list(factor_df.index.values)

    def test_make_joined_gene_specific_df(self):
        joined_df = self.jmsmd.make_joined_gene_specific_df(["ENSG00000003400"])
        assert joined_df is not None
        assert joined_df.index.name == self.jmsmd.patient_id_colname

    def test_make_joined_df(self):
        joined_df = self.jmsmd.make_joined_df()
        assert joined_df is not None
        assert joined_df.index.name == self.jmsmd.patient_id_colname


if __name__ == '__main__':
    unittest.main()
