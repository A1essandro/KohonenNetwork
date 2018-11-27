using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using NeuralNetworkConstructor.Learning.Samples;
using NeuralNetworkConstructor.Learning.Strategies;
using NeuralNetworkConstructor.Structure.Layers;
using NeuralNetworkConstructor.Structure.Nodes;
using NeuralNetworkConstructor.Structure.Summators;
using NeuralNetworkConstructor.Structure.Synapses;

namespace KohonenNetwork.Learning.Strategy
{
    public class UnsupervisedLearningVariableOutput : UnsupervisedLarningStrategyBase
    {

        private readonly double _criticalRange;
        private readonly int _maxNeurons;
        private readonly Func<IMasterNode, int, ISynapse> _synapseFactory;

        public UnsupervisedLearningVariableOutput(double criticalRange, int maxOutputNeurons = int.MaxValue)
        {
            _criticalRange = criticalRange;
            _maxNeurons = maxOutputNeurons;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public override async Task LearnSample(KohonenNetwork network, ISelfLearningSample sample, double theta)
        {
            Contract.Requires(network.OutputLayer as ILayer<INotInputNode> != null,
                $"OutputLayer of network must implements {nameof(ILayer<INotInputNode>)}");

            network.Input(sample.Input);
            var needCreateNeuron = await _needNewNeuron(network).ConfigureAwait(false);
            if (needCreateNeuron)
            {
                await _createNode(network).ConfigureAwait(false);
            }
            else
            {
                await _recalcWeights(network, theta).ConfigureAwait(false);
            }
        }

        #region private methods

        private async Task _createNode(KohonenNetwork network)
        {
            var index = 0;
            var newNode = new Neuron();
            ((ILayer<INotInputNode>)network.OutputLayer).AddNode(newNode);
            foreach (var inputNode in network.InputLayer.Nodes.OfType<IMasterNode>())
            {
                var synapse = _synapseFactory(inputNode, index++);
                newNode.AddSynapse(new Synapse(inputNode, await inputNode.Output()));
            }
        }

        private async Task<bool> _needNewNeuron(KohonenNetwork network)
        {
            if (network.OutputLayer.Nodes.Count() >= _maxNeurons)
            {
                return false;
            }

            if (await _checkRangeAsync(network).ConfigureAwait(false))
            {
                return false;
            }

            return true;
        }

        private async Task<bool> _checkRangeAsync(KohonenNetwork network)
        {
            var index = await network.GetOutputIndex();
            var outputNodes = network.OutputLayer.Nodes.ToArray();
            var euclidRange = await EuclidRangeSummator
                                        .GetEuclidRange(outputNodes[index] as ISlaveNode)
                                        .ConfigureAwait(false);

            return euclidRange < _criticalRange;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private async Task _recalcWeights(KohonenNetwork network, double theta)
        {
            var output = await network.Output().ConfigureAwait(false);
            GetWinner(network, output, theta).Synapses.AsParallel().ForAll(async synapse =>
            {
                var nodeOutput = await synapse.MasterNode.Output().ConfigureAwait(false);
                synapse.ChangeWeight(theta * (nodeOutput - synapse.Weight));
            });
        }

        #endregion

    }
}