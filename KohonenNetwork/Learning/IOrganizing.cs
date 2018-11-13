using System.Collections.Generic;
using System.Threading.Tasks;

namespace KohonenNetwork.Learning
{
    public interface IOrganizing
    {

        Task<bool> Organize(IEnumerable<double> input);

    }
}