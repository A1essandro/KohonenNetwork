using NeuralNetwork.Learning.Samples;
using NeuralNetwork.Structure.Layers;
using NeuralNetwork.Structure.Nodes;
using NeuralNetwork.Structure.Synapses;
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Threading.Tasks;
using NodeFactory = System.Func<NeuralNetwork.Kohonen.IKohonenNetwork, NeuralNetwork.Structure.Nodes.ISlaveNode>;
using SynapseFactory = System.Func<NeuralNetwork.Structure.Nodes.IMasterNode, NeuralNetwork.Structure.Nodes.ISlaveNode, double, NeuralNetwork.Structure.Synapses.ISynapse>;

namespace NeuralNetwork.Kohonen.Learning.Strategy
{
    public class UnsupervisedLearningVariableOutput : UnsupervisedLarningStrategyBase
    {

        private static SynapseFactory DefaultSynapseFactory = (master, slave, w) => new Synapse(master, slave, w);
        private static NodeFactory DefaultNodeFactory = network => new Neuron();

        private readonly double _criticalRange;
        private readonly int _maxNeurons;
        private readonly SynapseFactory _synapseFactory;
        private readonly NodeFactory _nodeFactory;

        public UnsupervisedLearningVariableOutput(double criticalRange, int maxOutputNeurons = int.MaxValue, NodeFactory nodeFactory = null, SynapseFactory synapseFactory = null)
        {
            _criticalRange = criticalRange;
            _maxNeurons = maxOutputNeurons;
            _nodeFactory = nodeFactory ?? DefaultNodeFactory;
            _synapseFactory = synapseFactory ?? DefaultSynapseFactory;
        }

        public override async Task LearnSample(IKohonenNetwork network, ISelfLearningSample sample, double theta)
        {
            Contract.Requires(network.OutputLayer as ILayer<INotInputNode> != null,
                $"OutputLayer of network must implements {nameof(ILayer<INotInputNode>)}");

            await network.Input(sample.Input);
            var needCreateNeuron = await _needNewNeuron(network).ConfigureAwait(false);
            if (needCreateNeuron)
            {
                _createNode(network);
            }
            else
            {
                await _recalcWeights(network, theta);
            }
        }

        #region private methods

        private void _createNode(IKohonenNetwork network)
        {
            ISlaveNode newNode = _createAndAddNewNode(network);

            _generateSynapsesToNewNode(network, newNode);
        }

        private ISlaveNode _createAndAddNewNode(IKohonenNetwork network)
        {
            var newNode = _nodeFactory(network);

            var nodes = network.OutputLayer.Nodes.ToList();
            ((ILayer<INotInputNode>)network.OutputLayer).AddNode(newNode);
            network.OutputLayer = network.OutputLayer;
            return newNode;
        }

        private void _generateSynapsesToNewNode(IKohonenNetwork network, ISlaveNode newNode)
        {
            foreach(var inputNode in network.InputLayer.Nodes.OfType<IMasterNode>())
            {
                var synapse = _synapseFactory(inputNode, newNode, inputNode.LastCalculatedValue);

                network.AddSynapse(synapse);
            }
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
            var euclidRange = _getEuclidRange(outputNodes);

            return euclidRange < _criticalRange;
        }

        private async Task _recalcWeights(IKohonenNetwork network, double theta)
        {
            var output = await network.Output();
            var winner = GetWinner(network, output, theta);
            var synapses = network.Synapses.Where(s => s.SlaveNode == winner);

            foreach(var synapse in synapses)
            {
                var nodeOutput = synapse.MasterNode.LastCalculatedValue;
                synapse.ChangeWeight(theta * (nodeOutput - synapse.Weight));
            }
        }

        private double _getEuclidRange(IEnumerable<INode> nodes)
        {
            var sum = nodes.Select(x => Math.Pow(x.LastCalculatedValue, 2)).Sum();

            return Math.Sqrt(sum);
        }

        #endregion

    }
}