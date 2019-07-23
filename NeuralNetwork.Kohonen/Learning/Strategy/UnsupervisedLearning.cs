using NeuralNetwork.Learning.Samples;
using System.Linq;
using System.Threading.Tasks;

namespace NeuralNetwork.Kohonen.Learning.Strategy
{
    public class UnsupervisedLearning : UnsupervisedLarningStrategyBase
    {

        public override Task LearnSample(IKohonenNetwork network, ISelfLearningSample sample, double theta)
        {
            network.Input(sample.Input);

            return _recalcWeights(network, theta);
        }

        #region private methods

        private async Task _recalcWeights(IKohonenNetwork network, double theta)
        {
            var output = await network.Output().ConfigureAwait(false);
            var winner = GetWinner(network, output, theta);
            var synapses = network.Synapses.Where(s => s.SlaveNode == winner);

            foreach (var synapse in synapses)
            {
                var nodeOutput = synapse.MasterNode.LastCalculatedValue;
                synapse.ChangeWeight(theta * (nodeOutput - synapse.Weight));
            }
        }

        #endregion

    }
}