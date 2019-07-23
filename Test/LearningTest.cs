using NeuralNetwork.Kohonen;
using NeuralNetwork.Kohonen.Learning;
using NeuralNetwork.Kohonen.Learning.Strategy;
using NeuralNetwork.Learning;
using NeuralNetwork.Learning.Samples;
using NeuralNetwork.Structure.Layers;
using NeuralNetwork.Structure.Networks;
using NeuralNetwork.Structure.Nodes;
using NeuralNetwork.Structure.Synapses;
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

            var network = new KohonenNetwork(inputLayer, outputLayer);
            LinkEachToEachNodes(network, inputLayer, outputLayer);

            return network;
        }

        [Fact]
        public async Task UnsupervisedLearningTest()
        {
            var inputLayer = new InputLayer(() => new InputNode(), 3);
            var outputLayer = new Layer(() => new Neuron(), 5);

            var network = new KohonenNetwork(inputLayer, outputLayer);
            LinkEachToEachNodes(network, inputLayer, outputLayer);

            var strategy = new UnsupervisedLearning();
            var learning = new Learning<KohonenNetwork, ISelfLearningSample>(network, strategy, new LearningSettings
            {
                EpochRepeats = 200,
                ThetaFactorPerEpoch = x => 0.975
            });

            var inputs = _getInputs().Select(x => new SelfLearningSample(x));
            await learning.Learn(inputs);

            await network.Input(_control[0]);
            var res0 = await network.GetOutputIndex();
            await network.Input(_control[1]);
            var res1 = await network.GetOutputIndex();
            await network.Input(_control[2]);
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
                synapseFactory: (n1, n2, w) => new Synapse(n1, n2, w));

            var learning = new Learning<KohonenNetwork, ISelfLearningSample>(network, strategy, new LearningSettings
            {
                EpochRepeats = 200,
                ThetaFactorPerEpoch = i => 0.975
            });

            var inputs = _getInputs().Select(x => new SelfLearningSample(x));
            await learning.Learn(inputs);

            await network.Input(_control[0]);
            var res0 = await network.GetOutputIndex();
            await network.Input(_control[1]);
            var res1 = await network.GetOutputIndex();
            await network.Input(_control[2]);
            var res2 = await network.GetOutputIndex();

            Assert.Equal(OUTPUT_QTY, network.OutputLayer.NodesQuantity);
            Assert.NotEqual(res0, res1);
            Assert.Equal(res1, res2);
        }

        private void LinkEachToEachNodes(ISimpleNetwork network, ILayer<IMasterNode> layer1, ILayer<INotInputNode> layer2)
        {
            var rand = new Random();
            foreach (var node1 in layer1.Nodes)
            {
                foreach (var node2 in layer2.Nodes)
                {
                    var weight = 2 * (rand.NextDouble() - 0.5);
                    var synapse = new Synapse(node1, node2 as ISlaveNode, weight);

                    network.AddSynapse(synapse);
                }
            }
        }

    }
}