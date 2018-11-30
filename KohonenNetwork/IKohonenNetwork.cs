using System.Collections.Generic;
using System.Threading.Tasks;
using NeuralNetworkConstructor.Networks;

namespace KohonenNetwork
{
    public interface IKohonenNetwork : INetwork
    {

        Task<IEnumerable<double>> RawOutput();

        Task<int?> GetOutputIndex();

    }
}