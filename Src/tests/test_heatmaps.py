import unittest
from heatmaps import Heatmaps


class HeatmapsTests(unittest.TestCase):

    def setUp(self):
        """ We test with factors (metagenes) and expression matrix from Factors/Mini_AOCS and
        metadata from Data/Mini_AOCS, both of which are comitted.  Note
        that the metadata rows (patients/samples) are deliberately NOT in the same order
        as in the columns of the expression matrix.   The code has to deal with this."""

        self.hm = Heatmaps('Mini_AOCS', 'Mini_AOCS', saveplots=False)
        self.combined_df = self.hm.make_combined_df()
        assert self.combined_df is not None
        assert self.combined_df.index.name == self.hm.patient_id_colname

    def test_init(self):
        assert self.hm is not None

    def test_plot_heatmap(self):
        self.hm.plot_heatmap(self.combined_df, show=False)

    def test_read_metadata_and_x_dfs(self):
        meta_df, x_df = self.hm.read_metadata_and_x_dfs()
        print(meta_df.columns)
        print(x_df.columns)
        # It is crucial that these dataframes are aligned with respect to patients!
        assert list(meta_df.index.values) == list(x_df.columns)

    def test_read_factors_df(self):
        meta_df, x_df = self.hm.read_metadata_and_x_dfs()
        factor_df = self.hm.read_factors_df(x_df, 'ICA', 5)
        # print(factor_df)
        # print(meta_df)
        assert list(meta_df.index.values) == list(factor_df.index.values)


if __name__ == '__main__':
    unittest.main()
