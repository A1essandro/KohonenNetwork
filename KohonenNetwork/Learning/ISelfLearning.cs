using System.Collections.Generic;
using NeuralNetworkConstructor.Network;

namespace KohonenNetwork.Learning
{
    public interface ISelfLearning
    {

        void Learn(IEnumerable<double> input);

        void SetNetwork(INetwork network);

    }
}