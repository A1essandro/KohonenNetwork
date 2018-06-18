using System.Collections.Generic;
using System.Threading.Tasks;

namespace KohonenNetwork.Learning
{
    public interface IOrganizing
    {

        bool Organize(IEnumerable<double> input);

        Task<bool> OrganizeAsync(IEnumerable<double> input);

    }
}