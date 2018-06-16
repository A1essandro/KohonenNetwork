using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetworkConstructor.Network;
using NeuralNetworkConstructor.Network.Node;
using NeuralNetworkConstructor.Network.Node.ActivationFunction;

namespace KohonenNetwork.Learning
{

    /// <summary>
    /// Class for learning network
    /// </summary>
    public class SelfLearning : ISelfLearning
    {

        public const double DEFAULT_FORCE = 0.15;

        private INetwork _network;
        private readonly double _force;
        private readonly IOrganizing _organizing;

        public readonly bool _needOrganize;

        public SelfLearning(double force = DEFAULT_FORCE, IOrganizing organizingAlgorithm = null)
        {
            _force = force;
            _organizing = organizingAlgorithm;
            _needOrganize = organizingAlgorithm != null;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input">Input data for learning</param>
        /// <param name="force">Force of learning</param>
        public void Learn(IEnumerable<double> input)
        {
            if (_needOrganize && _organizing.Organize(input))
            {
                return;
            }

            _network.Input(input);
            var output = _network.Output();

            var winnerIndex = Array.IndexOf(output.ToArray(), output.Max());
            var winner = _network.OutputLayer.Nodes[winnerIndex];

            foreach (var synapse in (winner as ISlaveNode).Synapses)
            {
                var nodeOutput = synapse.MasterNode.Output();
                synapse.ChangeWeight(_force * (nodeOutput - synapse.Weight));
            }
        }

        public void SetNetwork(INetwork network)
        {
            _network = network;
        }

    }
}