using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using NeuralNetworkConstructor.Learning.Samples;
using NeuralNetworkConstructor.Learning.Strategies;
using NeuralNetworkConstructor.Structure.Nodes;

namespace KohonenNetwork.Learning.Strategy
{
    public class UnsupervisedLearning : ILearningStrategy<TwoLayersNetwork, ISelfLearningSample>
    {

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Task LearnSample(TwoLayersNetwork network, ISelfLearningSample sample, double theta)
        {
            network.Input(sample.Input);
            return _recalcWeights(network, theta);
        }

        #region private methods

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private async Task _recalcWeights(TwoLayersNetwork network, double theta)
        {
            var output = await network.Output().ConfigureAwait(false);
            _getWinner(network, output, theta).Synapses.AsParallel().ForAll(async synapse =>
            {
                var nodeOutput = await synapse.MasterNode.Output().ConfigureAwait(false);
                synapse.ChangeWeight(theta * (nodeOutput - synapse.Weight));
            });
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private ISlaveNode _getWinner(TwoLayersNetwork network, IEnumerable<double> output, double theta)
        {
            var winnerIndex = Array.IndexOf(output.ToArray(), output.Max());
            return network.OutputLayer.Nodes.ToArray()[winnerIndex] as ISlaveNode;
        }

        #endregion

    }
}