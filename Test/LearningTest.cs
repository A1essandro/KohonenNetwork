using NeuralNetwork.Kohonen;
using NeuralNetwork.Kohonen.Learning;
using NeuralNetwork.Kohonen.Learning.Strategy;
using NeuralNetwork.Learning;
using NeuralNetwork.Learning.Samples;
using NeuralNetwork.Structure.Layers;
using NeuralNetwork.Structure.Nodes;
using NeuralNetwork.Structure.Synapses;
using NeuralNetworkConstructor.Constructor.Generators;
using System;
using System.Linq;
using System.Threading.Tasks;
using Xunit;

namespace Test
{
    public class LearningTest
    {

        private readonly double[][] _control = new double[][]
            {
                new double[] {0.99, 0.49, 0.01},
                new double[] {0.03, 0.47, 0.99},
                new double[] {0.04, 0.55, 0.96}
            };

        private double[][] _getInputs()
        {
            var random = new Random();

            var inputs = new double[2500][];
            for (var i = 0; i < 2500; i++)
            {
                inputs[i] = new double[] { random.NextDouble(), random.NextDouble(), random.NextDouble() };
            }

            return inputs;
        }

        private KohonenNetwork _getNetwork()
        {
            var inputLayer = new InputLayer(() => new InputNode(), 3);
            var outputLayer = new Layer(() => new Neuron(), 5);

            new EachToEachSynapseGenerator<Synapse>().Generate(inputLayer, outputLayer);

            return new KohonenNetwork(inputLayer, outputLayer);
        }

        [Fact]
        public async Task UnsupervisedLearningTest()
        {
            var inputLayer = new InputLayer(() => new InputNode(), 3);
            var outputLayer = new Layer(() => new Neuron(), 5);
            new EachToEachSynapseGenerator<Synapse>().Generate(inputLayer, outputLayer);
            var network = new KohonenNetwork(inputLayer, outputLayer);

            var strategy = new UnsupervisedLearning();
            var learning = new Learning<KohonenNetwork, ISelfLearningSample>(network, strategy, new LearningSettings
            {
                EpochRepeats = 200,
                ThetaFactorPerEpoch = x => 0.975
            });

            var inputs = _getInputs().Select(x => new SelfLearningSample(x));
            await learning.Learn(inputs);

            network.Input(_control[0]);
            var res0 = await network.GetOutputIndex();
            network.Input(_control[1]);
            var res1 = await network.GetOutputIndex();
            network.Input(_control[2]);
            var res2 = await network.GetOutputIndex();

            Assert.NotEqual(res0, res1);
            Assert.Equal(res1, res2);
        }

        [Fact]
        public async Task UnsupervisedLearningVariableOutputTest()
        {
            const int OUTPUT_QTY = 5;
            var inputLayer = new InputLayer(() => new InputNode(), 3);
            var outputLayer = new Layer(); //without nodes
            var network = new KohonenNetwork(inputLayer, outputLayer);

            var strategy = new UnsupervisedLearningVariableOutput(
                criticalRange: 0.15, 
                maxOutputNeurons: OUTPUT_QTY, 
                synapseFactory: (n, w) => new Synapse(n, w));
            var learning = new Learning<KohonenNetwork, ISelfLearningSample>(network, strategy, new LearningSettings
            {
                EpochRepeats = 200,
                ThetaFactorPerEpoch = i => 0.975
            });

            var inputs = _getInputs().Select(x => new SelfLearningSample(x));
            await learning.Learn(inputs);

            network.Input(_control[0]);
            var res0 = await network.GetOutputIndex();
            network.Input(_control[1]);
            var res1 = await network.GetOutputIndex();
            network.Input(_control[2]);
            var res2 = await network.GetOutputIndex();

            Assert.Equal(OUTPUT_QTY, network.OutputLayer.NodesQuantity);
            Assert.NotEqual(res0, res1);
            Assert.Equal(res1, res2);
        }

    }
}