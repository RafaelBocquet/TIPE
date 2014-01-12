template<unsigned IB, unsigned FB>
FixedPoint<IB, FB> FixedPoint<IB, FB>::exp() const{
  // exp(exp_table_1[i]) = 2^(i);
  // exp(exp_table_1[IB-1]) = 2^(IB);
  static constexpr SelfType exp_table_1 [32] = {
    SelfType(     0                                                         ),
    SelfType(     0.6931471805599452862267639829951804131269454956054687    ),
    SelfType(     1.3862943611198905724535279659903608262538909912109375    ),
    SelfType(     2.07944154167983574765798948646988719701766967773437      ),
    SelfType(     2.77258872223978114490705593198072165250778198242187      ),
    SelfType(     3.46573590279972654215612237749155610799789428710937      ),
    SelfType(     4.15888308335967149531597897293977439403533935546875      ),
    SelfType(     4.85203026391961689256504541845060884952545166015625      ),
    SelfType(     5.54517744447956228981411186396144330501556396484375      ),
    SelfType(     6.23832462503950768706317830947227776050567626953125      ),
    SelfType(     6.93147180559945308431224475498311221599578857421875      ),
    SelfType(     7.62461898615939848156131120049394667148590087890625      ),
    SelfType(     8.317766166719342990631957945879548788070678710937        ),
    SelfType(     9.010913347279288387881024391390383243560791015625        ),
    SelfType(     9.704060527839233785130090836901217699050903320312        ),
    SelfType(     10.397207708399179182379157282412052154541015625          ),
    SelfType(     11.0903548889591245796282237279228866100311279296875      ),
    SelfType(     11.78350206951906997687729017343372106552124023437        ),
    SelfType(     12.4766492500790153741263566189445555210113525390625      ),
    SelfType(     13.16979643063896077137542306445538997650146484375        ),
    SelfType(     13.8629436111989061686244895099662244319915771484375      ),
    SelfType(     14.55609079175885156587355595547705888748168945312        ),
    SelfType(     15.2492379723187969631226224009878933429718017578125      ),
    SelfType(     15.942385152878742360371688846498727798461914062          ),
    SelfType(     16.63553233343868598126391589175909757614135742187        ),
    SelfType(     17.32867951399863315486982173752039670944213867187        ),
    SelfType(     18.02182669455857677576204878278076648712158203125        ),
    SelfType(     18.71497387511852394936795462854206562042236328125        ),
    SelfType(     19.40812105567846757026018167380243539810180664062        ),
    SelfType(     20.10126823623841474386608751956373453140258789062        ),
    SelfType(     20.7944154167983583647583145648241043090820312            ),
    SelfType(     21.4875625973583055383642204105854034423828125            )
  };
  // exp(exp_table_2[i]) = 2^(-i-1) + 1;
  static constexpr SelfType exp_table_2 [31] = {
    SelfType(     0.4054651081081643848591511414269916713237762451171875                           ),
    SelfType(     0.223143551314209764857565687634632922708988189697265625                         ),
    SelfType(     0.1177830356563834557359626842298894189298152923583984375                        ),
    SelfType(     0.06062462181643483993820353816772694699466228485107421875                       ),
    SelfType(     0.03077165866675368732785500469617545604705810546875                             ),
    SelfType(     0.015504186535965254478686148331689764745533466339111328125                      ),
    SelfType(     0.007782140442054948960282079184480608091689646244049072265625                   ),
    SelfType(     0.0038986404156573228885207527127931825816631317138671875                        ),
    SelfType(     0.0019512201312617493374756971746819544932805001735687255859375                  ),
    SelfType(     0.00097608597305545892475198144211390172131359577178955078125                    ),      
    SelfType(     0.0004881620795013511871808520314885981861152686178684234619140625               ),
    SelfType(     0.00024411082752736270738876112051940481251222081482410430908203125              ),
    SelfType(     0.000122062862525677370720104952805940001780982129275798797607421875             ),
    SelfType(     0.0000610332936806385270025153422235320022082305513322353363037109375            ),
    SelfType(     0.00003051711247318637969315295588312864083491149358451366424560546875           ),
    SelfType(     0.000015258672648362397970280522618846674731685197912156581878662109375          ),
    SelfType(     0.000007629365427567572438224442754606258176863775588572025299072265625          ),
    SelfType(     0.00000381468998968588967482334049774461703918859711848199367523193359375        ),
    SelfType(     0.000001907346813825409489364238850572785821668730932287871837615966796875       ),
    SelfType(     9.536738616591882694082499792587181985936695127747952938079833984375E-7          ),
    SelfType(     4.7683704451632343610290617856584116651674776221625506877899169921875E-7         ),
    SelfType(     2.38418550679857595928304455322466193223363006836734712123870849609375E-7        ),
    SelfType(     1.192092824453544614992595777717976357479301441344432532787322998046875E-7       ),
    SelfType(     5.960464299903385839164343998862471973865240215673111379146575927734375E-8       ),
    SelfType(     2.9802321943606112576104734466879431220576179839554242789745330810546875E-8      ),
    SelfType(     1.49011610828253554418455710636921829337353528899257071316242218017578125E-8     ),
    SelfType(     7.4505805691682525093710864894092082977294921875E-9                              ),
    SelfType(     3.725290291523020158592771622352302074432373046875E-9                            ),
    SelfType(     1.86264514749623355527319290558807551860809326171875E-9                          ),
    SelfType(     9.313225741817976466307982263970188796520233154296875E-10                        ),
    SelfType(     4.65661287199319040563949556599254719913005828857421875E-10                      )
  };
  std::int32_t x = mValue;
  bool neg = x < 0; if(neg){ x = -x; }
  if(x >= exp_table_1[IB].mValue){
    return SelfType(maximum);
  }
  SelfType result;
  unsigned lo = 0, hi = IB;
  while(hi != lo){
    unsigned mid = (hi + lo + 1) / 2;
    if(x < exp_table_1[mid].mValue){
      hi = mid - 1;
    }else{
      lo = mid;
    }
  }
  result.mValue = unit << lo;
  x -= exp_table_1[lo].mValue;
  for(unsigned i = 0; i < 31; ++i){
    while(x >= exp_table_2[i].mValue && exp_table_2[i].mValue != 0){
      x -= exp_table_2[i].mValue;
      result.mValue += result.mValue >> (i + 1);
    }
  }
  return neg ? SelfType(1.0) / result : result;
}