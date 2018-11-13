using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using NeuralNetworkConstructor.Structure.ActivationFunctions;
using NeuralNetworkConstructor.Structure.Nodes;
using NeuralNetworkConstructor.Structure.Summators;
using NeuralNetworkConstructor.Structure.Synapses;

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
        
        public async Task<bool> Organize(IEnumerable<double> input)
        {
            if (_network.OutputLayer.Nodes.Count >= _maxNeurons)
            {
                return false;
            }

            if (await _checkRangeAsync(input).ConfigureAwait(false))
            {
                return false;
            }

            await _createNode();

            return true;
        }

        #region Private methods

        private async Task<bool> _checkRangeAsync(IEnumerable<double> input)
        {
            await _network.Input(input);
            var index = await _network.GetOutputIndex();
            var euclidRange = await EuclidRangeSummator
                                        .GetEuclidRange(_network.OutputLayer.Nodes[index] as ISlaveNode)
                                        .ConfigureAwait(false);

            return euclidRange < _criticalRange;
        }

        private async Task _createNode()
        {
            var newNode = new Neuron();
            _network.OutputLayer.Nodes.Add(newNode);
            foreach (INode inputNode in _network.InputLayer.Nodes)
            {
                newNode.AddSynapse(new Synapse(inputNode, await inputNode.Output()));
            }
        }

        #endregion

    }
}