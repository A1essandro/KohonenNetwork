using System;
using System.Collections.Generic;
using System.Linq;
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

        /// <summary>
        /// 
        /// </summary>
        /// <param name="learning"></param>
        /// <param name="criticalRange">Critical range for decide to start training or add a new neuron</param>
        public Organizing(KohonenNetwork<TFunc> network, double criticalRange)
        {
            _network = network;
            _criticalRange = criticalRange;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input">Input data for checking</param>
        public bool Organize(IEnumerable<double> input)
        {
            _network.Input(input);
            var index = _network.GetOutputIndex();
            var outputLayerNodes = _network.OutputLayer.Nodes;
            var euclidRange = EuclidRangeSummator.GetEuclidRange(outputLayerNodes[index] as ISlaveNode);
            if (euclidRange < _criticalRange)
            {
                return false;
            }

            var newNode = new Neuron<TFunc>();
            _network.OutputLayer.Nodes.Add(newNode);
            foreach (INode inputNode in _network.InputLayer.Nodes)
            {
                newNode.AddSynapse(new Synapse(inputNode, inputNode.Output()));
            }

            return true;
        }

    }
}