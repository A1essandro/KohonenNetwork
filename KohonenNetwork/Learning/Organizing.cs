using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using NeuralNetworkConstructor.Network.Node;
using NeuralNetworkConstructor.Network.Node.ActivationFunction;
using NeuralNetworkConstructor.Network.Node.Summator;
using NeuralNetworkConstructor.Network.Node.Synapse;

namespace KohonenNetwork.Learning
{

    /// <summary>
    /// Class for self-organizing of network
    /// </summary>
    public class Organizing<TFunc> : IOrganizing
        where TFunc : IActivationFunction, new()
    {

        private readonly KohonenNetwork<TFunc> _network;
        private readonly double _criticalRange;
        private readonly int _maxNeurons;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="learning"></param>
        /// <param name="criticalRange">Critical range for decide to start training or add a new neuron</param>
        public Organizing(KohonenNetwork<TFunc> network, double criticalRange, int maxOutputNeurons = int.MaxValue)
        {
            _network = network;
            _criticalRange = criticalRange;
            _maxNeurons = maxOutputNeurons;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input">Input data for checking</param>
        public bool Organize(IEnumerable<double> input)
        {
            if (_network.OutputLayer.Nodes.Count >= _maxNeurons)
            {
                return false;
            }

            if (_checkRange(input))
            {
                return false;
            }

            _createNode();

            return true;
        }

        public async Task<bool> OrganizeAsync(IEnumerable<double> input)
        {
            if (_network.OutputLayer.Nodes.Count >= _maxNeurons)
            {
                return false;
            }

            if (await _checkRangeAsync(input).ConfigureAwait(false))
            {
                return false;
            }

            _createNode();

            return true;
        }

        #region Private methods

        private bool _checkRange(IEnumerable<double> input)
        {
            _network.Input(input);
            var index = _network.GetOutputIndex();
            var euclidRange = EuclidRangeSummator.GetEuclidRange(_network.OutputLayer.Nodes[index] as ISlaveNode);

            return euclidRange < _criticalRange;
        }

        private async Task<bool> _checkRangeAsync(IEnumerable<double> input)
        {
            _network.Input(input);
            var index = await _network.GetOutputIndexAsync();
            var euclidRange = await EuclidRangeSummator
                                        .GetEuclidRangeAsync(_network.OutputLayer.Nodes[index] as ISlaveNode)
                                        .ConfigureAwait(false);

            return euclidRange < _criticalRange;
        }

        private void _createNode()
        {
            var newNode = new Neuron<TFunc>();
            _network.OutputLayer.Nodes.Add(newNode);
            foreach (INode inputNode in _network.InputLayer.Nodes)
            {
                newNode.AddSynapse(new Synapse(inputNode, inputNode.Output()));
            }
        }

        #endregion

    }
}