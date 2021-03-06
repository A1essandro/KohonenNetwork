﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using KohonenNetwork.Learning;
using NeuralNetworkConstructor.Constructor;
using NeuralNetworkConstructor.Network;
using NeuralNetworkConstructor.Network.Layer;
using NeuralNetworkConstructor.Network.Node;
using NeuralNetworkConstructor.Network.Node.ActivationFunction;
using NeuralNetworkConstructor.Network.Node.Synapse;

namespace KohonenNetwork
{
    public class KohonenNetwork<TFunc> : TwoLayersNetwork
        where TFunc : IActivationFunction, new()
    {

        public KohonenNetwork(int inputNodes, int outputNodes, bool withBias = false)
            : this(new NetworkConfiguration(inputNodes, outputNodes, withBias))
        {
        }

        public KohonenNetwork(NetworkConfiguration config)
            : base(CreateAndGetInputLayer(config.InputLayerNodes, config.CreateBiasNode), CreateAndGetOutputLayer(config.OutputLayerNodes))
        {
            Synapse.Generator.EachToEach(InputLayer, OutputLayer, config.SynapseWeightGenerator);
        }

        public override IEnumerable<double> Output() => _prepareResult(base.Output());

        public override async Task<IEnumerable<double>> OutputAsync() => _prepareResult(await base.OutputAsync().ConfigureAwait(false));

        public IEnumerable<double> RawOutput() => Output();

        public async Task<IEnumerable<double>> RawOutputAsync() => await OutputAsync().ConfigureAwait(false);

        public int GetOutputIndex()
        {
            var output = Output();

            return Array.IndexOf(output.ToArray(), output.Max());
        }

        public async Task<int> GetOutputIndexAsync()
        {
            var output = await OutputAsync().ConfigureAwait(false);

            return Array.IndexOf(output.ToArray(), output.Max());
        }

        #region Private methods

        private double[] _prepareResult(IEnumerable<double> raw)
        {
            var winnerIndex = Array.IndexOf(raw.ToArray(), raw.Max());
            var result = new double[_outputLayer.Nodes.Count];
            result[winnerIndex] = 1;

            return result;
        }

        private static InputLayer CreateAndGetInputLayer(int qty, bool withBias)
        {
            var result = new InputLayer();
            for (var i = 0; i < qty; i++)
            {
                result.Nodes.Add(new InputNode());
            }

            if (withBias)
            {
                result.Nodes.Add(new InputBias());
            }

            return result;
        }

        private static Layer CreateAndGetOutputLayer(int qty)
        {
            var result = new Layer();
            for (var i = 0; i < qty; i++)
            {
                result.Nodes.Add(new Neuron<TFunc>());
            }

            return result;
        }

        #endregion

    }
}
