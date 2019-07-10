using NeuralNetwork.Learning.Samples;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

namespace NeuralNetwork.Kohonen.Learning.Strategy
{
    public class UnsupervisedLearning : UnsupervisedLarningStrategyBase
    {

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public override Task LearnSample(IKohonenNetwork network, ISelfLearningSample sample, double theta)
        {
            network.Input(sample.Input);
            return _recalcWeights(network, theta);
        }

        #region private methods

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private async Task _recalcWeights(IKohonenNetwork network, double theta)
        {
            var output = await network.Output().ConfigureAwait(false);
            var recalcTasks = GetWinner(network, output, theta).Synapses.Select(async synapse =>
            {
                var nodeOutput = await synapse.MasterNode.Output().ConfigureAwait(false);
                synapse.ChangeWeight(theta * (nodeOutput - synapse.Weight));
            });

            await Task.WhenAll(recalcTasks);
        }

        #endregion

    }
}