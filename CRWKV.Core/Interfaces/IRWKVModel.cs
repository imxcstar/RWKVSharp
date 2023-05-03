using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RWKV
{
    public interface IRWKVModel
    {
        public object GetEmptyStates();

        public (IEnumerable<float> logits, object state) Forward(int token, object state);
    }
}
