using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace RWKVSharp.Core.Tokenizer
{
    public interface ITokenizer
    {
        public List<int> Encode(string text);

        public string Decode(IEnumerable<int> tokens);
    }
}
