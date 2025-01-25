
static const int _ = []() { std::ios::sync_with_stdio(false); std::cin.tie(nullptr); std::cout.tie(nullptr); return 0; }();


// finding all descedant of a node u


set<int> dfs(int u , int p , vector<int> adj[]) {
    set<int> s;
    s.insert(u);
    for(auto v: adj[u]) {
        if(v == p) continue;
        set<int> t = dfs(v , u , Query , adj , P , ans);
        if(s.size() < t.size()) swap(s , t);
        for(auto &i: t) s.insert(i);  
    }
    
    return s;
}



// dijakstra

vector<int> dijakstra(vector<pair<int,int>> adj[] , int n , int source) {
    vector<int> dist(n , 1e18);
    dist[source] = 0;
 
    // priority_queue<pair<int,int> , vector<pair<int,int>> , greater<pair<int,int>>> pq;
    multiset<pair<int,int>> pq;
    pq.insert({0 , source});
 
    while(!pq.empty()) {
        pair<int,int> pr = *pq.begin();
        pq.erase(pq.begin());
 
        int node = pr.second;
        int dists = pr.first;
 
        for(auto child: adj[node]) {
            int newdist = dist[node] + child.second;
            if(newdist < dist[child.first]) {
                if(dist[child.first] != 1e18) {
                    pq.erase({dist[child.first] , child.first});
                }
                dist[child.first] = newdist;
                pq.insert({newdist , child.first});
            }
        } 
    }
    return dist;

}
// Zero one BFS

void bfs_zero_one() {
    int n , m;
    cin >> n >> m;

    vector<pair<int, int>> adj[n];

    for (int i = 0; i < m; i++) {
        int u , v;
        cin >> u >> v;
        u--; v--;
        if (u == v) continue;
        adj[u].push_back({v , 0});
        adj[v].push_back({u , 1});
    }

    vector<int> dist(n , 1e18);
    dist[0] = 0;

    deque<int> dq;

    dq.push_back(0);

    while (!dq.empty()) {
        int u = dq.front();
        dq.pop_front();

        for (auto V : adj[u]) {
            int v = V.first;
            int w = V.second;

            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                if (w == 1) {
                    dq.push_back(v);
                }
                else {
                    dq.push_front(v);
                }
            }
        }
    }
    cout << dist[n - 1] << endl;
}

// xor from a vertices to all other nodes

void dfs(int u , int p , vector<pair<int,int>> adj[] , vector<int>& a , int& curxor) {

    a[u] = curxor;

    for(auto child: adj[u]) {
        int v = child.first;
        int wt = child.second;

        if(v == p) continue;

        curxor = curxor ^ wt;
        dfs(v , u , adj , a , curxor);
        curxor = curxor ^ wt;
    }
}


// pattern finding in the string


bool sub(string pat, string txt){
    int M = pat.size();
    int N = txt.size();
    int lps[M];
    auto compute = [&](){
        int len = 0;
        lps[0] = 0;
        int i = 1;
        while (i < M){
            if (pat[i] == pat[len]) {
                len++;
                lps[i] = len;
                i++;
            }
            else{
                if (len != 0){
                    len = lps[len - 1];
                }
                else{
                    lps[i] = 0;
                    i++;
                }
            }
        }
    };
    compute();
    int i = 0,j = 0;
    while (i < N) {
        if (pat[j] == txt[i]){
            j++;
            i++;
        }
        if (j == M){
            //Found pattern at index (i - j)
            return true;
            j = lps[j - 1];
        }
        else if (i < N && pat[j] != txt[i]) {
            if (j != 0)j = lps[j - 1];
            else i = i + 1;
        }
    }
    return false;
}


class Node {
public:
    Node* link[26];
    int cnt = 0;
 
    bool containsKey(char ch) {
        return link[ch - 'a'] != NULL;
    }
    void put(char ch) {
        link[ch - 'a'] =  new Node();
    }
    Node* get(char ch) {
        return link[ch - 'a'];
    }
};
 
class TRIE {
    Node* root;
public:
    TRIE() {
        root = new Node();
    }
    void insert(string& s) {
        int n = s.size();
        Node* node = root;
        for(int i = 0; i < n; i++) {
            if(!node -> containsKey(s[i])) {
                node -> put(s[i]);
            }
            node = node -> get(s[i]);
            node -> cnt++;
        }
    }
 
    int helper(string& s , int val) {
        int n = s.size();
        Node* node = root;
        for(int i = n - 1; i >= 0; i--) {
            if(node -> containsKey(s[i])) {
                node = node -> get(s[i]);
                val = val - node -> cnt * 2;
                debug(node -> cnt)
            }
            else {
                break;
            }
        }
        return val;
    }
};


// range maximum and minimum

struct rmq {
    vector<vector<int>> st;
    int n;
    int sign;
    rmq() {}
    rmq(vector<int> &v , int s) {
        n = v.size();
        sign = s;
        st = vector<vector<int>>(20, vector<int>(n));
        for (int j = 0; j < n; ++j) st[0][j] = v[j];
        for (int i = 0; (1 << i) <= n; ++i) {
            for (int j = 0; j + (1 << i+1) <= n; ++j) {
                st[i+1][j] = update(st[i][j], st[i][j + (1 << i)]);
            }
        }
    }
    
    int query(int l, int r) {
        if (l > r) return -1;
        int j = l == r ? 0 : 31 - __builtin_clz(r - l);
        return update(st[j][l], st[j][r - (1 << j) + 1]);
    }

    int update(int a , int b) {
        if(sign == 0) {
            return min(a , b);
        }
        return max(a , b);
    }
};


// Finding the centroid of a tree

int dfs(int u , int p ,  vector<int> adj[] , vector<int>& dp , vector<int>& centroid , int n) {
    int cnt = 1;
    bool ok = true;
    for(auto v: adj[u]) {
        if(v == p) continue;
        int temp = dfs(v , u , adj , dp , centroid , n);
        if(temp > n / 2) {
            ok = false;
        }
        cnt += temp;
    }
    dp[u] = cnt;
    if(n - dp[u] > n / 2) {
        ok = false;
    }
    if(ok) {
        centroid.push_back(u);
    }
    return dp[u];
}
 


// Give an index of element in the original array for the sorted array 

vector<int> indPos(vector<int>& a) {
    int n = a.size();
    vector<int> b(n);
    iota(b.begin() , b.end() , 0);

    sort(b.begin() , b.end() , [&](int x , int y) {
        return a[x] < a[y];
    });
    return b;
}


void shortest_distance(vector<vector<int>>&matrix){
        int n = matrix.size();
        
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                if(i == j) {
                    matrix[i][j] = 0;
                }
                else if(matrix[i][j] == -1) {
                    matrix[i][j] = 1e9; // inf distance
                }
            }
        }
        
        for(int node = 0; node < n; node++) {
            for(int i = 0; i < n; i++) {
                for(int j = 0; j < n; j++) {
                    matrix[i][j] = min(matrix[i][j] , matrix[i][node] + matrix[node][j]);
                }
            }
        }
        
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                if(matrix[i][j] == 1e9) {
                    matrix[i][j] = -1;
                }
            }
        }
    }

// Kth ancestor

void solve() {
    int n , q;
    cin >> n >> q;

    vector<int> p(n , -1);
    for(int i = 1; i < n; i++) {
        int curp;
        cin >> curp;
        curp--;
        p[i] = curp;
    }

    const int MAX = 20;
    vector<vector<int>> Kancestor(MAX , vector<int>(n , -1));

    for(int i = 0; i < n; i++) {
        Kancestor[0][i] = p[i];
    }

    for(int i = 1; i < MAX; i++) {
        for(int j = 0; j < n; j++) {
            int f = Kancestor[i - 1][j];
            if(f != -1) {
                f = Kancestor[i - 1][f];
            }
            Kancestor[i][j] = f;
        }
    }

    for(int i = 0; i < q; i++) {
        int node , k;
        cin >> node >> k;
        node--;
        for(int bit = 0; bit < MAX; bit++) {
            int curbit = ((k >> bit) & 1);
            if(curbit == 1) {
                node = Kancestor[bit][node];
                if(node == -1) {
                    break;
                }
            }
        }
        if(node == -1) node--;
        cout << node + 1 << endl;
    }
}


void dfs(int u , int p , int level , vector<int>& depth, vector<int> adj[]) {
    depth[u] = level;
    for(auto v: adj[u]) {
        if(p != v) {
            dfs(v , u , level + 1 , depth , adj);
        }
    }
}
 

// lca of nodes in log time with processing

 
class LCA {
public:
    const int MAX = 18;
    vector<vector<int>> Kancestor;
    vector<vector<int>> adj;
    int S = 0;
    vector<int> depth;
    LCA(int n , vector<int>& p , vector<int> adjs[] , int Src) { // ith parent of i is intially i
        S = Src;
        depth.resize(n , 0);
        for(int i = 0; i < n; i++) {
            adj.push_back(adjs[i]);
        }
        Kancestor.resize(MAX , vector<int>(n , -1));
        for(int i = 0; i < n; i++) {
             Kancestor[0][i] = p[i];
        }
 
        for(int i = 1; i < MAX; i++) {
            for(int j = 0; j < n; j++) {
                Kancestor[i][j] = Kancestor[i - 1][Kancestor[i - 1][j]];
                
            }
        }
     
        auto kth = [&](int node , int k) {
            for(int bit = 0; bit < MAX; bit++) {
                int curbit = ((k >> bit) & 1);
                if(curbit == 1) {
                    node = Kancestor[bit][node];
                }
            }
            return node;
        };
 
        int level = 0;
        dfs(S , -1 , 0 , depth , adj);
    }
 
    void dfs(int u , int p , int level , vector<int>& depth, vector<vector<int>>& adj) {
        depth[u] = level;
        for(auto v: adj[u]) {
            if(p != v) {
                dfs(v , u , level + 1 , depth , adj);
            }
        }
    }
 
    int kth(int node , int k) {
        for(int bit = 0; bit < MAX; bit++) {
            int curbit = ((k >> bit) & 1);
            if(curbit == 1) {
                node = Kancestor[bit][node];
            }
        }
        return node;
    }
 
    int query(int u , int v) { // u and v are 0 - based indexing
        int du = depth[u];
        int dv = depth[v];
        if(du > dv){ 
            swap(u , v);
            swap(du , dv);
        }
        int diff = dv - du;
        v = kth(v , diff);
 
        if(u == v) {
            return u;
        }
        
        int ans = 0;

        for(int i = 17; i >= 0; i--) {
            if(Kancestor[i][u] != Kancestor[i][v]) {
                u = Kancestor[i][u];
                v = Kancestor[i][v];
            }
        }
        u = Kancestor[0][u];
        return u;
    }
};




void bitSet() {
    const int N = 5e4;
    bitset<N> bs;
    int i = 4;
    bs.set(i);
    bs.reset(i);
    bs.flip(i);
    bs.size();
    bs.count(); // number of set bits

//     operator&=
// operator|=
// operator^=
// operator~
 
// performs binary AND, OR, XOR and NOT
// (public member function)
// operator<<=
// operator>>=
// operator<<
// operator>>
 
// performs binary shift left and shift right
// (public member function)
 bs.all();
 bs.any();
 bs.none();
 bs.to_string();
}


class HashSet {
private:
    vector<vector<int>> v;
    int N;
public:
    HashSet(int n) {
        N = n;
        v.resize(n);
    }

    int generateHash(int x) {
        return x % N;
    }

    bool add(int x) {
        int pos = generateHash(x);
        if(tofind(x)) return false;
        v[pos].push_back(x);
        return true;
    }

    bool tofind(int x) {
        int pos = generateHash(x);
        return find(v[pos].begin() , v[pos].end() , x) != v[pos].end();
    }

    bool erase(int x) {
        int pos = generateHash(x);
        if(!tofind(x)) return false;
        v[pos].erase(find(v[pos].begin() , v[pos].end() , x));
    } 
};


// sum of unique character over all subtrings. Hopefilly it is correct
// https://leetcode.com/contest/weekly-contest-291/problems/total-appeal-of-a-string/

class Solution {
public:
    int uniqueLetterString(string s) {
        int n = s.size();
        const int N = 26;
        vector<vector<int>> a(N);
        for(int i = 0; i < n; i++) {
            a[s[i] - 'A'].push_back(i);
        }
        int ans = 0;
        set<int> st;
        for(int i = 0; i < 26; i++) st.insert(i);
        for(int i = 0; i < n; i++) {
            set<int> temp = st;
            int ind = i;
            int counter = 1;
            while(ind < n) {
                temp.erase(s[ind] - 'A');
                int maxi = n;
                for(auto v: temp) {
                    if(a[v].size() == 0 or a[v].back()<= ind) continue;
                    auto it = upper_bound(a[v].begin() , a[v].end() , ind);
                    int curind = *it;
                    maxi = min(maxi , curind);
                }
                ans += counter * (maxi - ind);
                counter += 1;
                ind = maxi;
            }
            // cout << ans << endl;
        }
        return ans;
    }
};

/*
    a  b  c
    1  3  3 + 3 -> 10
    
    a  b  a
    1  3  3 + 2

    l e e t c o d e
    1 2 

*/




class SH {
public:
    vector<int> Hash , power , invMod;
    int P , Mod;

    SH(string& s , int P , int Mod) {
        int n = s.size();
        Hash.assign(n + 1 , 0);
        power.assign(n + 1 , 0);
        invMod.assign(n + 1 , 0);
        this -> P = P;
        this -> Mod = Mod;
        power[0] = 1;
        for(int i = 1; i <= n; i++) {
            power[i] = (power[i - 1] * P) % Mod;
        }
        for(int i = 0; i < n; i++) {
            invMod[i] = exp(power[i] , Mod - 2);
        }
        for(int i = 0; i < n; i++) {
            Hash[i + 1] = (Hash[i] + (s[i] - 'a' + 1ll) * power[i]) % Mod;
        }

    }

    int exp(int a , int b) {
        int ans = 1;
        while(b > 0) {
            if(b & 1) {
                ans = ans * a;
                ans %= Mod;
            }
            a = a * a;
            a %= Mod;
            b >>= 1;    
        }
        return ans;
    }

    int getHash(int l , int r) {
        if(l > r) return -1;
        int curh = (Hash[r + 1] - Hash[l] + Mod) % Mod;
        curh = (curh * invMod[l]) % Mod;
        curh = (curh + Mod) % Mod;
        return curh;
    }
};  


struct Hash{
    int b, n; // b = number of hashes
    const int mod = 1e9 + 7;
    vector<vector<int>> fw, bc, pb, ib;
    vector<int> bases;
 
    inline int power(int x, int y){
        if (y == 0){
            return 1;
        }
 
        int v = power(x, y / 2);
        v = 1LL * v * v % mod;
        if (y & 1) return 1LL * v * x % mod;
        else return v;
    }
 
    inline void init(int nn, int bb, string str){
        n = nn;
        b = bb;
        fw = vector<vector<int>>(b, vector<int>(n + 2, 0));
        bc = vector<vector<int>>(b, vector<int>(n + 2, 0));
        pb = vector<vector<int>>(b, vector<int>(n + 2, 1));
        ib = vector<vector<int>>(b, vector<int>(n + 2, 1));
        bases = vector<int>(b);
        str = "0" + str;
 
        for (auto &x : bases) x = RNG() % (mod / 10);
 
        for (int i = 0; i < b; i++){
            for (int j = 1; j <= n + 1; j++){
                pb[i][j] = 1LL * pb[i][j - 1] * bases[i] % mod;
            }
            ib[i][n + 1] = power(pb[i][n + 1], mod - 2);
            for (int j = n; j >= 1; j--){
                ib[i][j] = 1LL * ib[i][j + 1] * bases[i] % mod;
            }
 
            for (int j = 1; j <= n; j++){
                fw[i][j] = (fw[i][j - 1] + 1LL * (str[j] - 'a' + 1) * pb[i][j]) % mod;
            }
            for (int j = n; j >= 1; j--){
                bc[i][j] = (bc[i][j + 1] + 1LL * (str[j] - 'a' + 1) * pb[i][n + 1 - j]) % mod;
            }
        }
    }
 
    inline int getfwhash(int l, int r, int i){
        int ans = fw[i][r] - fw[i][l - 1];
        ans = 1LL * ans * ib[i][l - 1] % mod;
        
        if (ans < 0) ans += mod;
 
        return ans;
    } 
 
    inline int getbchash(int l, int r, int i){
        int ans = bc[i][l] - bc[i][r + 1];
        ans = 1LL * ans * ib[i][n - r] % mod;
 
        if (ans < 0) ans += mod;
 
        return ans;
    }
 
    inline bool equal(int l1, int r1, int l2, int r2){
        for (int i = 0; i < b; i++){
            int v1, v2;
            if (l1 <= r1) v1 = getfwhash(l1, r1, i);
            else v1 = getbchash(r1, l1, i);
 
            if (l2 <= r2) v2 = getfwhash(l2, r2, i);
            else v2 = getbchash(r2, l2, i);
 
            if (v1 != v2) return false;
        }
        return true;
    }
 
    inline bool pal(int l, int r){
        return equal(l, r, r, l);
    }
};


 Hash h;
h.init(n, 2, s)


class DSU {
public:
    vector<int> parent , rank , size;
    DSU(int n) {
        parent.assign(n , 0);
        rank.assign(n , 0);
        size.assign(n , 1);
        iota(parent.begin() , parent.end() , 0);
    }

    int find(int u) {
        if(u == parent[u]) {
            return u;
        }
        return parent[u] = find(parent[u]);
    }

    int together(int u , int v) {
        return find(u) == find(v);
    }

    void unionByRank(int u , int v) {
        int up = find(u) , vp = find(v);
        if(up == vp) return;
        if(rank[up] > rank[vp]) {
            parent[vp] = up;
        } 
        else if(rank[up] < rank[vp]) {
            parent[up] = vp;
        }
        else {
            parent[up] = vp;
            rank[vp]++;
        }
    }
    void unionBySize(int u , int v) {
        int up = find(u) , vp = find(v);
        if(up == vp) return;
        if(size[up] > size[vp]) {
            size[up] += size[vp];
            parent[vp] = up;
        }
        else {
            size[vp] += size[up];
            parent[up] = vp;
        }
    }
    int count() {
        int ans = 0;
        int n = parent.size();
        for(int i = 0; i < n; i++) {
            if(i == parent[i]) {
                ans++;
            }
        }
        return ans;
    }
    int compSize(int u) {
        return size[parent[u]];
    }
};



// Deletion and addition in kanpsack

 vector<int> dp(1001,0);
    dp[0] = 1;
    const int mod = 1e9 + 7;
    int Q;
    cin >> Q;
    while(Q--) {
        int tt,x;
        cin >> tt >> x;
        if(tt == 0) {
            for(int i=1000;i>=x;--i)
                dp[i] = (dp[i] + dp[i-x]) % mod;
        }
        else if(tt == 1) {
            for(int i=x;i<=1000;++i)
                dp[i] = (dp[i] - dp[i-x] + mod) % mod;
        }
        else {
            cout << dp[x] << "\n";
        }
    }




// Or in the range

const int N = 31;

vector<vector<int>> a(N , vector<int>(n , 0));

for(int i = 0; i < n; i++) {
    for(int j = 0; j < N; j++) {
        a[j][i] = ((arr[i] >> j) & 1);
    }
}

for(int j = 0; j < N; j++) {
        for(int i = 1; i < n; i++) {
            a[j][i] += a[j][i - 1];
        }
}

auto rs = [&](int row , int l , int r) {
        if(l > r) return 0ll;
        if(l == 0) {
            return a[row][r];
        }
        return (a[row][r] - a[row][l - 1]);
};

auto rangeOr = [&](int l , int r) {
    int num = 0;
    for(int j = 0; j <= 30; j++) {
        int x = rs(j , l , r);
        if(x > 0) {
            num = (num | (1ll << j));
        }
    }
    return num;
};

#include <bits/stdc++.h>

using namespace std;
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

int Rand(int l, int h){
    return uniform_int_distribution<int>(l, h)(rng);
}
int main(){
    int n = Rand(1, 100); // length of the regular bracket is 2 * n
    stack<char> st;
    int pref = 0;
    for (int i = 1; i <= n; ++i){
        int x = Rand(0, 1);
        if (x == 1){
            pref ++;
            st.push('(');
            cout << '(';
        }
        else{
            if (pref > 0) {
                pref--; st.push(')');
                cout << ')';
            }
            else{
                pref++; st.push('(');
                cout << '(';
            }
        }
    }
    while(st.size()){
        char c = st.top(); st.pop();
        cout << (c == '(' ? ')' : '(');
    }
}


// get frequency in the range for queries

class getFreq {
public:
    vector<int> a;
    map<int,vector<int>> mp;
    int n;

    getFreq(vector<int>& a) {
        this -> a = a;
        this -> n = a.size();
        for(int i = 0; i < n; i++) {
            mp[a[i]].push_back(i);
        }
    }

    int getFrequency(int ele , int l , int r) {
        auto beginItr = mp[ele].begin();
        auto endItr = mp[ele].end();

        auto it = lower_bound(beginItr , endItr , l);;
       

        if(it == endItr) return 0;
        auto itr = upper_bound(beginItr , endItr , r);
        if(itr == beginItr) return 0;

        itr--;
        int firstInd = it - beginItr;
        int lastInd = itr - beginItr;

        if(lastInd < firstInd) return 0;
        return lastInd - firstInd + 1;
    }

};


void buildSparseTable(vector<int> &arr, int n, vector<vector<int>> &sparse)
{
    for (int i = 0; i < n; i++)
        sparse[i][0] = arr[i];
 
    /* Build sparse table*/
    for (int m = 1; m < 60; m++)
        for (int i = 0; i <= n - (1ll << m); i++)
 
            /* Updating the value of gcd. */
            sparse[i][m] = __gcd(sparse[i][m - 1],
                                 sparse[i + (1ll << (m - 1))][m - 1]);
}
 
/* Query Processed */
int query(int L, int R, vector<vector<int>> &sparse)
{
    /* The maximum power of range that is less than 2 */
    int m = (int)log2(R - L + 1);
    return __gcd(sparse[L][m], sparse[R - (1ll << m) + 1][m]);
}



int rand(int l, int r){
    static mt19937 
    rng(chrono::steady_clock::now().time_since_epoch().count());
    uniform_int_distribution<int> ludo(l, r); 
    return ludo(rng);
}


int main() {

}




