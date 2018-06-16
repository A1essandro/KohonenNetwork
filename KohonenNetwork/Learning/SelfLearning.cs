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
    public class SelfLearning<TFunc> : ISelfLearning
        where TFunc : IActivationFunction, new()
    {

        private readonly KohonenNetwork<TFunc> _network;
        private readonly IOrganizing _organizing;

        public readonly bool _needOrganize;

        public SelfLearning(KohonenNetwork<TFunc> network, IOrganizing organizingAlgorithm = null)
        {
            _network = network;
            _organizing = organizingAlgorithm;
            _needOrganize = organizingAlgorithm == null;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input">Input data for learning</param>
        /// <param name="force">Force of learning</param>
        public void Learn(ICollection<double> input, double force)
        {
            if (_needOrganize && _organizing.Organize(input))
            {
                return;
            }

            _network.Input(input);
            var output = _network.Output();

            var winnerIndex = Array.IndexOf(output.ToArray(), output.Max());
            var winner = _network.Layers.Last().Nodes[winnerIndex];

            foreach (var synapse in (winner as ISlaveNode).Synapses)
            {
                synapse.ChangeWeight(force * (synapse.MasterNode.Output() - synapse.Weight));
            }
        }

    }
}