"""Reproduce Chapter 2 and descriptive results from local NSL-KDD files. Python 3.11+; no network or fitting."""
from pathlib import Path
import json, re, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SHARED = ROOT.parent / '00_DU_LIEU_CHUNG'
PROCESSED = SHARED / 'data' / 'processed'
GROUPS = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
FEATURES = '''duration protocol_type service flag src_bytes dst_bytes land wrong_fragment urgent hot num_failed_logins logged_in num_compromised root_shell su_attempted num_root num_file_creations num_shells num_access_files num_outbound_cmds is_host_login is_guest_login count srv_count serror_rate srv_serror_rate rerror_rate srv_rerror_rate same_srv_rate diff_srv_rate srv_diff_host_rate dst_host_count dst_host_srv_count dst_host_same_srv_rate dst_host_diff_srv_rate dst_host_same_src_port_rate dst_host_srv_diff_host_rate dst_host_serror_rate dst_host_srv_serror_rate dst_host_rerror_rate dst_host_srv_rerror_rate'''.split()
CATS = ['protocol_type', 'service', 'flag']
NUM = [c for c in FEATURES if c not in CATS]
RATES = [c for c in NUM if c.endswith('_rate')]
BINARY = ['land','logged_in','root_shell','is_host_login','is_guest_login']
# UCI training_attack_types; test-only extensions follow Imrana et al. (2022), Table 3.
FAMILIES = {
 'Normal': ['normal'],
 'DoS': 'back land neptune pod smurf teardrop apache2 mailbomb processtable udpstorm worm'.split(),
 'Probe': 'ipsweep nmap portsweep satan mscan saint'.split(),
 'R2L': 'ftp_write guess_passwd imap multihop phf spy warezclient warezmaster sendmail named snmpgetattack snmpguess xlock xsnoop httptunnel'.split(),
 'U2R': 'buffer_overflow loadmodule perl rootkit ps sqlattack xterm'.split()
}
MAPPING = {a:g for g,aa in FAMILIES.items() for a in aa}
SELECT = ['duration','src_bytes','dst_bytes','count','srv_count','serror_rate']
COMPARE = ['src_bytes','count','serror_rate']
UNITS = {'duration':'giây','src_bytes':'byte','dst_bytes':'byte','count':'kết nối','srv_count':'kết nối','serror_rate':'tỷ lệ (0–1)'}

def save(df, name):
    df.to_csv(ROOT/'tables'/name, index=False, float_format='%.15g')

def label_data(d):
    unknown = set(d.attack_name) - set(MAPPING)
    if unknown:
        raise ValueError(f'Unknown attack names: {sorted(unknown)}')
    d = d.copy()
    d['attack_category'] = pd.Categorical(d.attack_name.map(MAPPING),categories=GROUPS,ordered=True)
    d['label_binary'] = (d.attack_name != 'normal').astype('int64')
    return d

def stats(s):
    q = s.quantile([.25,.5,.75],interpolation='linear')
    return dict(n=len(s),mean=s.mean(),median=q.loc[.5],variance=s.var(ddof=1),std=s.std(ddof=1),min=s.min(),max=s.max(),q1=q.loc[.25],q3=q.loc[.75],iqr=q.loc[.75]-q.loc[.25],zero_pct=100*s.eq(0).mean())

def audit(d, split, raw):
    num = d[NUM+['difficulty']]
    parsed = num.apply(pd.to_numeric,errors='coerce')
    types = int((parsed.isna() & num.notna()).sum().sum())
    repeated = d.groupby(FEATURES,dropna=False,observed=True).agg(rows=('attack_name','size'), labels=('attack_name','nunique'))
    dup = repeated[repeated.rows > 1]
    conflict = repeated[repeated.labels > 1]
    bad_binary = (~d[BINARY].isin([0,1])).sum().sum()
    su_bad = (~d.su_attempted.isin([0,1])).sum()
    domain = int((d[NUM] < 0).sum().sum() + ((d[RATES]<0)|(d[RATES]>1)).sum().sum() + bad_binary)
    out = dict(rows=len(d),fields=d.shape[1],missing_cells=int(d.isna().sum().sum()),blank_string_cells=int(raw.apply(lambda s:s.str.strip().eq('')).sum().sum()),nonfinite_cells=int((~np.isfinite(parsed.to_numpy(dtype=float)) & ~parsed.isna().to_numpy()).sum()),type_errors=types,duplicate43_extra=int(d.duplicated().sum()),duplicate41_extra=int(d.duplicated(FEATURES).sum()),duplicate41_groups=len(dup),duplicate41_involved=int(dup.rows.sum()),conflict_groups=len(conflict),conflict_rows=int(conflict.rows.sum()),negative_cells=int((d[NUM]<0).sum().sum()),rate_outside_cells=int(((d[RATES]<0)|(d[RATES]>1)).sum().sum()),binary5_invalid_cells=int(bad_binary),su_not_binary_rows=int(su_bad),constant_features=int(d[FEATURES].nunique().eq(1).sum()),near_constant_features=int(sum(d[c].value_counts(normalize=True).iloc[0]>=.99 and d[c].nunique()>1 for c in FEATURES)),domain_invalid_excluding_su=domain)
    if len(conflict):
        key = conflict.reset_index()[FEATURES]
        d.merge(key,on=FEATURES).to_csv(ROOT/'tables'/f'{split}_conflicting_rows.csv',index=False)
    assert out['missing_cells']==out['blank_string_cells']==out['nonfinite_cells']==out['type_errors']==0
    assert out['domain_invalid_excluding_su']==0
    assert d.su_attempted.isin([0,1,2]).all()
    return out

def main():
    for folder in ['tables','figures']:
        (ROOT/folder).mkdir(parents=True,exist_ok=True)
    PROCESSED.mkdir(parents=True,exist_ok=True)
    dfs={}; audits={}; schema=[]; details=[]; checksums={}
    arff = (ROOT/'data/raw/KDDTrain+.arff').read_text()
    assert re.findall(r"@attribute '([^']+)'",arff)[:41] == FEATURES
    allowed={}
    for c in CATS:
        body=re.search(r"@attribute '"+c+r"'\s+\{([^}]+)\}",arff).group(1)
        allowed[c]=re.findall("'([^']+)'",body)
    for split,stem,n in [('train','KDDTrain+',125973),('test','KDDTest+',22544)]:
        path=ROOT/'data/raw'/f'{stem}.txt';checksums[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        raw=pd.read_csv(path,header=None,dtype=str,keep_default_na=False)
        assert raw.shape==(n,43)
        d=pd.read_csv(path,header=None,names=FEATURES+['attack_name','difficulty'])
        audits[split]=audit(d,split,raw)
        for c in CATS:
            assert set(d[c]).issubset(allowed[c]), (split,c)
        for c in NUM+['difficulty']:
            assert pd.api.types.is_numeric_dtype(d[c])
            if c not in RATES:assert (d[c] % 1 == 0).all()
        assert d.difficulty.between(0,21).all()
        d=label_data(d);dfs[split]=d
        d.to_csv(PROCESSED/f'{split}_clean.csv',index=False,float_format='%.15g')
        for c in FEATURES+['attack_name','difficulty','attack_category','label_binary']:
            s=d[c]
            schema.append(dict(split=split,feature=c,dtype=str(s.dtype),unique=s.nunique(),minimum=s.min() if c in NUM else '',maximum=s.max() if c in NUM else '',mode=str(s.mode().iloc[0]),mode_pct=100*s.value_counts(normalize=True).iloc[0]))
        for c in NUM:
            st=stats(d[c]);lo=st['q1']-1.5*st['iqr'];hi=st['q3']+1.5*st['iqr']
            details.append(dict(split=split,feature=c,**st,iqr_flag_rows=int(((d[c]<lo)|(d[c]>hi)).sum()),iqr_degenerate=st['iqr']==0))
        for col in ['attack_category','label_binary']:
            counts=d[col].value_counts(sort=False).reindex(GROUPS if col=='attack_category' else [0,1],fill_value=0)
            assert counts.sum()==n
            t=counts.rename_axis(col).reset_index(name='count');t['percent']=100*t['count']/n
            assert np.isclose(t.percent.sum(),100)
            save(t,f'{split}_{col}_distribution.csv')
    save(pd.DataFrame(audits).rename_axis('criterion').reset_index(),'quality_audit.csv')
    save(pd.DataFrame(schema),'schema_audit.csv'); save(pd.DataFrame(details),'all_numeric_descriptive.csv')
    train=dfs['train'];test=dfs['test']
    rows=[]
    for g in GROUPS:
        for a in FAMILIES[g]:rows.append(dict(attack_name=a,attack_category=g,label_binary=int(g!='Normal'),train_count=int(train.attack_name.eq(a).sum()),test_count=int(test.attack_name.eq(a).sum()),source='UCI training_attack_types' if a in set(train.attack_name)-{'normal'} else ('literal normal' if a=='normal' else 'Imrana et al. 2022, Table 3')))
    save(pd.DataFrame(rows),'attack_mapping.csv')
    dd=[]
    definitions={'duration':'Thời lượng kết nối','src_bytes':'Số byte từ nguồn đến đích','dst_bytes':'Số byte từ đích về nguồn','count':'Số kết nối tới cùng máy đích trong hai giây trước','srv_count':'Số kết nối tới cùng dịch vụ trong hai giây trước','serror_rate':'Tỷ lệ kết nối cùng máy đích có lỗi SYN','protocol_type':'Loại giao thức','service':'Dịch vụ mạng ở đích','flag':'Trạng thái bình thường hoặc lỗi của kết nối','su_attempted':'Chỉ báo thử lệnh su root theo mô tả gốc; dữ liệu có 0/1/2, giữ nguyên','is_host_login':'Tên trường theo ARFF NSL-KDD; tài liệu KDD cũ dùng is_hot_login, chưa khẳng định tương đương'}
    definitions.update({
      'land':'1 nếu nguồn và đích cùng host/cổng; ngược lại 0',
      'wrong_fragment':'Số mảnh gói tin sai', 'urgent':'Số gói có cờ urgent',
      'hot':'Số chỉ báo hot trong nội dung kết nối', 'num_failed_logins':'Số lần đăng nhập thất bại',
      'logged_in':'1 nếu đăng nhập thành công; ngược lại 0',
      'num_compromised':'Số điều kiện compromised được ghi nhận',
      'root_shell':'1 nếu đã có root shell; ngược lại 0',
      'num_root':'Số truy cập root', 'num_file_creations':'Số thao tác tạo tệp',
      'num_shells':'Số shell prompt', 'num_access_files':'Số thao tác trên tệp kiểm soát truy cập',
      'num_outbound_cmds':'Số lệnh outbound trong phiên FTP',
      'is_guest_login':'1 nếu đăng nhập bằng tài khoản guest; ngược lại 0',
      'srv_serror_rate':'Tỷ lệ lỗi SYN trong các kết nối cùng dịch vụ ở cửa sổ hai giây',
      'rerror_rate':'Tỷ lệ lỗi REJ trong các kết nối cùng máy đích ở cửa sổ hai giây',
      'srv_rerror_rate':'Tỷ lệ lỗi REJ trong các kết nối cùng dịch vụ ở cửa sổ hai giây',
      'same_srv_rate':'Tỷ lệ cùng dịch vụ trong các kết nối cùng máy đích ở cửa sổ hai giây',
      'diff_srv_rate':'Tỷ lệ khác dịch vụ trong các kết nối cùng máy đích ở cửa sổ hai giây',
      'srv_diff_host_rate':'Tỷ lệ khác máy đích trong các kết nối cùng dịch vụ ở cửa sổ hai giây',
      'dst_host_count':'Số kết nối cùng máy đích trong cửa sổ lịch sử kết nối',
      'dst_host_srv_count':'Số kết nối cùng máy đích và dịch vụ trong cửa sổ lịch sử kết nối',
      'dst_host_same_srv_rate':'Tỷ lệ cùng dịch vụ trong cửa sổ kết nối theo máy đích',
      'dst_host_diff_srv_rate':'Tỷ lệ khác dịch vụ trong cửa sổ kết nối theo máy đích',
      'dst_host_same_src_port_rate':'Tỷ lệ cùng cổng nguồn trong cửa sổ kết nối theo máy đích',
      'dst_host_srv_diff_host_rate':'Tỷ lệ khác máy đích trong các kết nối cùng dịch vụ ở cửa sổ lịch sử',
      'dst_host_serror_rate':'Tỷ lệ lỗi SYN trong cửa sổ kết nối theo máy đích',
      'dst_host_srv_serror_rate':'Tỷ lệ lỗi SYN trong kết nối cùng dịch vụ ở cửa sổ theo máy đích',
      'dst_host_rerror_rate':'Tỷ lệ lỗi REJ trong cửa sổ kết nối theo máy đích',
      'dst_host_srv_rerror_rate':'Tỷ lệ lỗi REJ trong kết nối cùng dịch vụ ở cửa sổ theo máy đích'
    })
    for c in FEATURES+['attack_name','difficulty','attack_category','label_binary']:
        role='network_feature' if c in FEATURES else ('metadata_not_predictor' if c=='difficulty' else 'label_not_predictor')
        kind='nominal' if c in CATS+['attack_name','attack_category'] else ('binary' if c in BINARY+['label_binary'] else ('ratio' if c in RATES else 'integer'))
        unit=UNITS.get(c,'tỷ lệ (0–1)' if c in RATES else ('chỉ báo' if c in BINARY else ('không áp dụng' if kind=='nominal' or c not in FEATURES else 'số đếm / mã nguyên')))
        dd.append(dict(position=len(dd)+1,name=c,type=kind,unit=unit,role=role,description=definitions.get(c,{'difficulty':'Số bộ phân loại gốc dự đoán đúng, 0–21; không phải đặc trưng mạng','attack_name':'Nhãn gốc TXT bảo toàn','attack_category':'Normal, DoS, Probe, R2L, U2R','label_binary':'Normal=0; Attack=1'}.get(c,'Tên đặc trưng bảo toàn theo schema ARFF; xem mô tả KDD chính thức')),train_dtype=str(train[c].dtype),test_dtype=str(test[c].dtype)))
    pd.DataFrame(dd).to_csv(PROCESSED/'data_dictionary.csv',index=False)
    save(pd.DataFrame([dict(feature=c,unit=UNITS[c],**stats(train[c])) for c in SELECT]),'selected_descriptive.csv')
    comp=[]
    for c in COMPARE:
        for v,g in [(0,'Normal'),(1,'Attack')]:comp.append(dict(feature=c,group=g,**stats(train.loc[train.label_binary==v,c])))
    save(pd.DataFrame(comp),'normal_attack_comparison.csv')
    # Retain 43 original fields plus two derived labels; no removal, resampling or fitting.
    save(pd.DataFrame([dict(split=s,before_rows=len(d),after_rows=len(d),before_fields=43,after_fields=45,network_features=41,added_labels=2,missing_after=int(d.isna().sum().sum()),removed_rows=0) for s,d in dfs.items()]),'before_after.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    palette=['#267796','#c65b39']
    fig,ax=plt.subplots(figsize=(8.5,3.1),layout='constrained')
    counts=train.attack_category.value_counts().reindex(GROUPS)
    pct=counts/len(train)*100
    bars=ax.bar(GROUPS,pct,color=['#267796','#c65b39','#7c8f42','#8764a3','#c69a3b'])
    for b,v in zip(bars,pct):ax.text(b.get_x()+b.get_width()/2,v+1,f'{v:.3f}%',ha='center',fontsize=10)
    ax.set(ylabel='Tỷ lệ trong KDDTrain+ (%)',ylim=(0,62));ax.grid(axis='y',alpha=.15)
    fig.savefig(ROOT/'figures/task2_labels.pdf',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(10,3.2),layout='constrained')
    for ax,c in zip(axes,COMPARE):
        transform=c=='src_bytes'
        arrays=[train.loc[train.label_binary==v,c].to_numpy() for v in [0,1]]
        if transform:arrays=[np.log1p(a) for a in arrays]
        edges=np.linspace(0,max(a.max() for a in arrays),41)
        for a,g,color in zip(arrays,['Normal','Attack'],palette):ax.hist(a,bins=edges,weights=np.ones(len(a))*100/len(a),histtype='step',lw=1.6,label=g,color=color)
        ax.set(xlabel='ln(1 + src_bytes/byte)' if transform else c, ylabel='Tỷ lệ trong nhóm / khoảng (%)');ax.grid(alpha=.15)
    axes[0].legend(frameon=False);fig.savefig(ROOT/'figures/task2_histograms.pdf',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(10,3.3),layout='constrained')
    for ax,c in zip(axes,COMPARE):
        arrays=[train.loc[train.label_binary==v,c].to_numpy() for v in [0,1]]
        if c=='src_bytes':arrays=[np.log1p(a) for a in arrays]
        bp=ax.boxplot(arrays,tick_labels=['Normal','Attack'],whis=1.5,showfliers=True,patch_artist=True,flierprops=dict(marker='.',markersize=1.2,alpha=.22),medianprops=dict(color='black',linewidth=1.6))
        for box,color in zip(bp['boxes'],palette):box.set_facecolor(color);box.set_alpha(.6)
        ax.set_ylabel('ln(1 + src_bytes/byte)' if c=='src_bytes' else c);ax.grid(axis='y',alpha=.15)
    fig.savefig(ROOT/'figures/task2_boxplots.pdf',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
    (ROOT/'tables/audit.json').write_text(json.dumps(audits,ensure_ascii=False,indent=2))
    (ROOT/'tables/raw_sha256.json').write_text(json.dumps(checksums,indent=2))
    print(json.dumps(audits,indent=2));print('PASS: schema, domains (explicit su_attempted exception), mapping, counts, proportions. No rows changed.')

if __name__=='__main__':
    main()
    from full_scope_data import enrich
    enrich()
