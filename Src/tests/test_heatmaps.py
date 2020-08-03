import unittest
from heatmaps import Heatmaps
from join_metasamples_metadata import JoinMetasamplesMetadata


class HeatmapsTests(unittest.TestCase):

    def setUp(self):
        self.hm = Heatmaps('Mini_AOCS', 'Mini_AOCS', saveplots=False)
        jmsmd = JoinMetasamplesMetadata('Mini_AOCS', 'Mini_AOCS')
        self.joined_df = jmsmd.make_joined_df()

    def test_init(self):
        assert self.hm is not None

    def test_plot_heatmap(self):
        self.hm.plot_heatmap(self.joined_df, show=False)


if __name__ == '__main__':
    unittest.main()
