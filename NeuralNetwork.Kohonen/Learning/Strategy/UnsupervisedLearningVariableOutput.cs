using NeuralNetwork.Structure.Layers;
using NeuralNetwork.Structure.Nodes;
using NeuralNetwork.Structure.Summators;
using NeuralNetwork.Structure.Synapses;
using NeuralNetworkConstructor.Learning.Samples;
using System;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

namespace NeuralNetwork.Kohonen.Learning.Strategy
{
    public class UnsupervisedLearningVariableOutput : UnsupervisedLarningStrategyBase
    {

        private static Func<IMasterNode, double, ISynapse> DefaultSynapseFactory = (n, w) => new Synapse(n, w);

        private readonly double _criticalRange;
        private readonly int _maxNeurons;
        private readonly Func<IMasterNode, double, ISynapse> _synapseFactory;

        public UnsupervisedLearningVariableOutput(double criticalRange, int maxOutputNeurons = int.MaxValue, Func<IMasterNode, double, ISynapse> synapseFactory = null)
        {
            _criticalRange = criticalRange;
            _maxNeurons = maxOutputNeurons;
            _synapseFactory = synapseFactory ?? DefaultSynapseFactory;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public override async Task LearnSample(IKohonenNetwork network, ISelfLearningSample sample, double theta)
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

        private Task _createNode(IKohonenNetwork network)
        {
            var newNode = new Neuron();
            ((ILayer<INotInputNode>)network.OutputLayer).AddNode(newNode);

            var tasks = network.InputLayer.Nodes.OfType<IMasterNode>().Select(async inputNode => {
                var synapse = _synapseFactory(inputNode, await inputNode.Output());
                newNode.AddSynapse(synapse);
            });

            return Task.WhenAll(tasks);
        }

        private async Task<bool> _needNewNeuron(IKohonenNetwork network)
        {
            if (network.OutputLayer.NodesQuantity >= _maxNeurons)
            {
                return false;
            }

            if (await _checkRangeAsync(network).ConfigureAwait(false))
            {
                return false;
            }

            return true;
        }

        private async Task<bool> _checkRangeAsync(IKohonenNetwork network)
        {
            var index = await network.GetOutputIndex();
            if (!index.HasValue)
                return false;
            var outputNodes = network.OutputLayer.Nodes.ToArray();
            var euclidRange = await EuclidRangeSummator
                                        .GetEuclidRange(outputNodes[index.Value] as ISlaveNode)
                                        .ConfigureAwait(false);

            return euclidRange < _criticalRange;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private async Task _recalcWeights(IKohonenNetwork network, double theta)
        {
            var output = await network.Output().ConfigureAwait(false);
            var recalcTasks = GetWinner(network, output, theta).Synapses.Select(async synapse =>
            {
                var nodeOutput = await synapse.MasterNode.Output().ConfigureAwait(false);
                synapse.ChangeWeight(theta * (nodeOutput - synapse.Weight));
            });

            await Task.WhenAll(recalcTasks).ConfigureAwait(false);
        }

        #endregion

    }
}