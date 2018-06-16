using KohonenNetwork;
using KohonenNetwork.Learning;
using NeuralNetworkConstructor.Network.Node.ActivationFunction;
using Xunit;
using System;

namespace Test
{
    public class LearningTest
    {

        [Fact]
        public void Learning()
        {
            var network = new KohonenNetwork<Logistic>(3, 2, false);
            network.SetLearning(new SelfLearning(0.5));
            var random = new Random();

            var control = new double[][]
            {
                new double[] {1, 0.5, 0},
                new double[] {0, 0.4, 1},
                new double[] {0, 0.7, 0.9}
            };

            for (var i = 0; i < 25000; i++)
            {
                network.Learn(new double[] { random.NextDouble(), random.NextDouble(), random.NextDouble() });
            }

            network.Input(control[0]);
            var res0 = network.GetOutputIndex();
            network.Input(control[1]);
            var res1 = network.GetOutputIndex();
            network.Input(control[2]);
            var res2 = network.GetOutputIndex();

            Assert.True(res0 != res1);
            Assert.True(res1 == res2);
        }

    }
}