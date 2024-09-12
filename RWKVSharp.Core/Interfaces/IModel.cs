using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RWKVSharp.Core.Interfaces
{
    public interface IModel
    {
        public object GetEmptyStates();

        public object GetStates(int[] tokens);

        public (IEnumerable<float> logits, object state) Forward(int token, object state);

        public (IEnumerable<float> logits, object state) Forward(IEnumerable<int> tokens, object state);
    }
}
